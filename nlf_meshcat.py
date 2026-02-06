import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

import yaml
def load_intrinsics(yaml_path):
    with open(yaml_path, 'r') as f:
        # On lit le texte et on supprime les balises qui font planter PyYAML
        content = f.read()
        content = content.replace("%YAML:1.0", "")
        content = content.replace("!!opencv-matrix", "")
        
        data = yaml.safe_load(content)
    
    # Dans ton fichier, K est un dictionnaire avec 'data'
    k_data = data['K']['data']
    # On transforme la liste de 9 chiffres en matrice 3x3
    K = np.array(k_data, dtype=np.float32).reshape(3, 3)
    return torch.from_numpy(K).unsqueeze(0) # Prêt pour NLF [1, 3, 3]
    

def load_extrinsics_for_nlf(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    ext = data['camera_extrinsics']
    R = np.array(ext['rotation_matrix'], dtype=np.float32)
    T = np.array(ext['translation_vector'], dtype=np.float32)

    M_cam_to_world = np.eye(4, dtype=np.float32)
    M_cam_to_world[:3, :3] = R
    M_cam_to_world[:3, 3] = T

    # NLF a besoin de World -> Camera 
    M_world_to_cam = np.linalg.inv(M_cam_to_world)

    return torch.from_numpy(M_world_to_cam).unsqueeze(0)

def load_camera_to_world(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    ext = data['camera_extrinsics']
    # On utilise float64 ici pour éviter l'erreur UnsupportedTypeException
    R = np.array(ext['rotation_matrix'], dtype=np.float64)
    T = np.array(ext['translation_vector'], dtype=np.float64)
    
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = T
    return M

def process_video(video_path, yolo_model_path, nlf_model_path, output_path="output_video.mp4"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    intrinsics = load_intrinsics("/home/kchalabi/Documents/4279/intrinsics/camera_0_intrinsics.yaml").to(device)
    #extrinsics = load_extrinsics_for_nlf("/home/kchalabi/Documents/4279/extrinsics/cam_to_world/camera_0/camera_0_extrinsics.yaml").to(device)

    # --- 1. INITIALISATION MESHCAT ---
    vis = meshcat.Visualizer()
    #vis = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000", web_url="http://127.0.0.1:7000")

    print(f"\n>>> Meshcat visualizer disponible ici : {vis.url()}")
    # Création d'un groupe pour les marqueurs 3D
    vis_markers = vis["markers"]
    M_cam_to_world = load_camera_to_world("/home/kchalabi/Documents/4279/extrinsics/cam_to_world/camera_0/camera_0_extrinsics.yaml")
    vis_markers.set_transform(M_cam_to_world)

    # --- 2. CHARGEMENT DES MODÈLES ---
    yolo = YOLO(yolo_model_path)
    nlf = torch.jit.load(nlf_model_path).eval().to(device)

    # Préparer les poids canoniques
    cano_verts_full = np.load("pipeline_nlf/canonical_verts/smplx.npy")
    indices = [5484, 6629, 3878, 7040, 4302, 7105, 4369, 7584, 4848, 7457, 4721,#c7, rshoulder,lshoulder,r_lelbow,l_lelbow,r_melbow,l_melbow,r_lwrist, l_lwrist,r_mwrist, l_mwrist
               8421, 5727, 8371, 5677, #r_asis,l_asis,r_psis,l_psis
               6401, 3640, 6407, 3646, 8576,5882,8680,8892, #r_knee,l_knee,r_mknee,l_mknee,r_ankle,l_ankle,r_mankle,l_mankle,
               8596,5902,8589,5895,8482,5788,8846,8634,#r_5meta, l_5meta, r_toe, l_toe, r_big_toe, l_big_toe, l_calc, r_calc,
               7978,4807,8004,5268,7483,4747,7664,4928,7776,5040,7887,5151,7420,4684,8078,5342, #7978,4807, r_tpinky, l_tpinky, r_bindex, l_bindex, r_tindex, l_tindex, r_tmiddle, l_tmiddle, r_tring,l_tring, r_bthumb, l_bthumb,r_tthumb, l_tthumb
               9008,9002,1253,399,10049,9503 ,  #nose, head,right_ear,left_ear, right_eye, left_eye
5941,5489,5500] #L2, T11, T6


    selected_points = torch.from_numpy(cano_verts_full[indices]).float().to(device)
    all_points = torch.from_numpy(cano_verts_full).float().to(device)
    weights = nlf.get_weights_for_canonical_points(all_points)

    # --- 3. PRÉPARATION VIDÉO ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la vidéo.")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Paramètres Caméra
    #focal_length = width / (2 * np.tan(np.deg2rad(55) / 2))
    #intrinsics = torch.tensor([[[focal_length, 0, width/2], 
      #                          [0, focal_length, height/2], 
     #                           [0, 0, 1]]]).float().to(device)
    extrinsics = torch.eye(4).unsqueeze(0).to(device)

    print(f"Début du traitement : {total_frames} frames...")

    frame_count = 0
    try:
        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret:
                break
            
            frame_start_time = time.time()

            # --- ETAPE A: YOLO ---
            yolo_results = yolo.predict(frame_bgr, classes=0, conf=0.25, device=device, verbose=False)[0]
            
            nlf_boxes = []
            if len(yolo_results.boxes) > 0:
                boxes = yolo_results.boxes.xyxy
                scores = yolo_results.boxes.conf.unsqueeze(1)
                wh = boxes[:, 2:] - boxes[:, :2]
                boxes_formatted = torch.cat([boxes[:, :2], wh, scores], dim=1)
                nlf_boxes.append(boxes_formatted)
            else:
                nlf_boxes.append(torch.zeros((0, 5)).to(device))

            # --- ETAPE B: NLF ---
            img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().divide(255).unsqueeze(0).to(device)

            with torch.inference_mode():
                outputs = nlf.estimate_poses_batched(
                    img_tensor, nlf_boxes,
                    intrinsic_matrix=intrinsics,
                    extrinsic_matrix=extrinsics,
                    world_up_vector=torch.tensor([0.0, -1.0, 0.0]).to(device),
                    weights=weights,
                    num_aug=1
                )

            poses_3d = outputs['poses3d'][0] # [Nb_personnes, Nb_points, 3]
            poses_3d = poses_3d/1000
            # --- ETAPE C: VISUALISATION MESHCAT ---
            if poses_3d.shape[0] > 0:
                # On récupère tous les points de toutes les personnes détectées
                # On convertit en [3, N] pour Meshcat
                points_all = poses_3d.view(-1, 3).cpu().numpy().T
                
                # Couleur : on crée un dégradé ou une couleur fixe (jaune ici)
                colors = np.zeros_like(points_all)
                colors[0, :] = 1.0  # R
                colors[1, :] = 0.0  # G
                colors[2, :] = 0.0  # B

                vis_markers.set_object(
                    g.PointCloud(position=points_all, color=colors, size=0.015)
                )
            else:
                # Si personne n'est détecté, on vide la scène (optionnel)
                vis_markers.delete()

            # --- ETAPE D: DESSIN 2D ET SAUVEGARDE ---
            processed_frame = draw_projections(frame_bgr, poses_3d, intrinsics)
            
            total_time = (time.time() - frame_start_time) * 1000
            cv2.putText(processed_frame, f"Total: {total_time:.1f}ms", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            out.write(processed_frame)
            frame_count += 1
            
            if frame_count % 10 == 0:
                print(f"Frame {frame_count}/{total_frames} traitée ({total_time:.1f} ms/frame)")

    finally:
        cap.release()
        out.release()
        print(f"\nTraitement terminé. Vidéo sauvegardée : {output_path}")

def draw_projections(img_bgr, poses_3d, intrinsics):
    img_viz = img_bgr.copy()
    poses_3d_np = poses_3d.cpu().numpy()
    K = intrinsics[0].cpu().numpy()

    for person_id in range(poses_3d_np.shape[0]):
        points_3d = poses_3d_np[person_id]
        # Taille point : petit pour le mesh (SMPL), gros pour les markers (20-50 pts)
        radius = 1 if len(points_3d) > 100 else 5
        color = (0, 255, 255) if len(points_3d) > 100 else (0, 0, 255)

        for pt_3d in points_3d:
            x, y, z = pt_3d
            if z > 0.1: # Éviter division par zéro ou points derrière caméra
                u = int((K[0, 0] * x / z) + K[0, 2])
                v = int((K[1, 1] * y / z) + K[1, 2])
                if 0 <= u < img_viz.shape[1] and 0 <= v < img_viz.shape[0]:
                    cv2.circle(img_viz, (u, v), radius, color, -1)
    return img_viz

if __name__ == "__main__":
    # Chemins à adapter
    video_in = "/home/kchalabi/Documents/StraightWalking/camera_0.mp4"
    yolo_p   = "/home/kchalabi/Documents/yolo11m.pt"
    nlf_p    = "/home/kchalabi/Documents/weights/nlf/nlf_s_multi_0.2.2.torchscript"
    
    process_video(video_in, yolo_p, nlf_p, "resultat_3d_meshcat.mp4")
