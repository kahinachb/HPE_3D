import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time
import smplfitter.pt # Assurez-vous d'avoir installé smplfitter
import os
import torch
import torch.nn as nn

# Hack pour compatibilité PyTorch ancienne version
if not hasattr(nn, 'Buffer'):
    nn.Buffer = lambda x: x

def process_video(video_path, yolo_model_path, nlf_model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Charger les modèles
    yolo = YOLO(yolo_model_path)
    nlf = torch.jit.load(nlf_model_path).eval().to(device)

    # 2. Préparer les points (indices pour focaliser le Fit)
    body_model_name = 'smpl'
    cano_verts_full = np.load(f"pipeline_nlf/canonical_verts/{body_model_name}.npy")
   # indices = [5621, 6629, 3878, 7040, 4302, 7105, 4369, 7584, 4848, 7457,
    #           4721, 8421, 5727, 8371, 5677, 6401, 3640, 6407, 3646, 8576,
     #          5882, 8680, 8892, 8596, 5902, 8589, 5895, 8482, 5788, 8846, 8634]

   # selected_points = torch.from_numpy(cano_verts_full[indices]).float().to(device)
    # On calcule les poids NLF uniquement pour ces indices
    #weights = nlf.get_weights_for_canonical_points(selected_points)
    all_points = torch.from_numpy(cano_verts_full).float().to(device)
    weights = nlf.get_weights_for_canonical_points(all_points)

    # 3. Ouvrir la vidéo
    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Configurer les deux sorties
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_raw = cv2.VideoWriter("resultat_raw.mp4", fourcc, fps, (width, height))
    out_fit = cv2.VideoWriter("resultat_fit.mp4", fourcc, fps, (width, height))

    # Paramètres Caméra (Forcer Float32 pour éviter l'erreur Double)
    focal_length = width / (2 * np.tan(np.deg2rad(55) / 2))
    intrinsics = torch.tensor([[[focal_length, 0, width/2], 
                                [0, focal_length, height/2], 
                                [0, 0, 1]]], dtype=torch.float32).to(device)
    extrinsics = torch.eye(4, dtype=torch.float32).unsqueeze(0).to(device)

    all_frames = []
    all_poses_3d = []

    print(f"--- Étape 1 : Inférence NLF (Raw) sur {total_frames} frames ---")
    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret: break
        
        # YOLO
        yolo_results = yolo.predict(frame_bgr, classes=0, conf=0.25, device=device, verbose=False)[0]
        
        # Formatage des boîtes (Forcer Float32)
        nlf_boxes = []
        if len(yolo_results.boxes) > 0:
            boxes = yolo_results.boxes.xyxy.float()
            scores = yolo_results.boxes.conf.unsqueeze(1).float()
            wh = boxes[:, 2:] - boxes[:, :2]
            nlf_boxes.append(torch.cat([boxes[:, :2], wh, scores], dim=1))
        else:
            nlf_boxes.append(torch.zeros((0, 5), dtype=torch.float32).to(device))

        # NLF
        img_tensor = torch.from_numpy(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().divide(255).unsqueeze(0).to(device)

        with torch.inference_mode():
            outputs = nlf.estimate_poses_batched(
                img_tensor, nlf_boxes,
                intrinsic_matrix=intrinsics,
                extrinsic_matrix=extrinsics,
                world_up_vector=torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32).to(device),
                weights=weights,
                num_aug=1
            )
        
        poses_3d = outputs['poses3d'][0] # [N_personnes, 31, 3]
        
        # Sauvegarde RAW
        img_raw = draw_projections(frame_bgr, poses_3d, intrinsics, color=(0, 0, 255))
        out_raw.write(img_raw)
        
        # Stockage pour le FIT
        all_frames.append(frame_bgr)
        all_poses_3d.append(poses_3d)
        
        if len(all_frames) % 50 == 0:
            print(f"Frame {len(all_frames)}/{total_frames} traitée...")

    # --- Étape 2 : SMPL Fit ---
    print("--- Étape 2 : Fitting SMPL (Lissage) ---")
    data_root = "/home/kchalabi/Documents/pipeline_nlf"
    os.environ['DATA_ROOT'] = data_root
    os.environ['BODY_MODEL_PATH'] = os.path.join(data_root, "body_models")

    # Hack pour le nom de fichier spécifique (basicmodel...) 
    # Si la lib râle encore sur le nom du fichier .pkl :
    smpl_dir = os.path.join(data_root, "body_models", "smpl")
    os.makedirs(smpl_dir, exist_ok=True)
    
    src = os.path.join(data_root, "smpl", "SMPL_NEUTRAL.pkl")
    dst = os.path.join(smpl_dir, "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl")
    
    if os.path.exists(src) and not os.path.exists(dst):
        import shutil
        shutil.copy(src, dst)
        print(f"Lien créé pour le modèle : {dst}")
    # Hack pour le fichier kid_template.npy manquant
    kid_template_path = os.path.join(smpl_dir, 'kid_template.npy')
    if not os.path.exists(kid_template_path):
        print("Fichier kid_template.npy manquant, création d'un fichier factice...")
        # On crée un template de la même taille que SMPL (6890 sommets)
        dummy_template = np.zeros((6890, 3), dtype=np.float32)
        np.save(kid_template_path, dummy_template)

    # On prépare la fonction de Fit
    # share_beta=True permet de garder la même morphologie pour la personne sur toute la vidéo
    fit_fn = smplfitter.pt.get_cached_fit_fn(
        body_model_name=body_model_name,
        num_betas=10,
        vertex_subset=None,
        share_beta=True, 
        device=device,
    )

    # Note: Cette boucle simplifiée assume 1 personne principale pour le fit
    for i in range(len(all_frames)):
        frame = all_frames[i]
        raw_pose = all_poses_3d[i] # [N, 31, 3]

        if raw_pose.shape[0] > 0:
            # On prend la première personne détectée pour le fit
            target = raw_pose[:1]
            with torch.inference_mode():
                # On ajuste le modèle SMPL-X aux 31 points
                fit_res = fit_fn(target,None, None,None)
            
            # Le résultat contient le maillage complet (par exemple 6890 sommets)
            # On projette le maillage "fitté" en jaune
            img_fit = draw_projections(frame, fit_res['vertices'], intrinsics, color=(0, 255, 255))
        else:
            img_fit = frame
            
        out_fit.write(img_fit)

    cap.release()
    out_raw.release()
    out_fit.release()
    print("Traitement terminé. Vidéos : resultat_raw.mp4 et resultat_fit.mp4")

def draw_projections(img_bgr, poses_3d, intrinsics, color=(0, 0, 255)):
    img_viz = img_bgr.copy()
    poses_np = poses_3d.cpu().numpy()
    K = intrinsics[0].cpu().numpy()

    for person in poses_np:
        # Si bcp de points (Full Mesh), petits points. Sinon gros points.
        radius = 1 if len(person) > 100 else 4
        for pt in person:
            if pt[2] != 0:
                u = int((K[0, 0] * pt[0] / pt[2]) + K[0, 2])
                v = int((K[1, 1] * pt[1] / pt[2]) + K[1, 2])
                if 0 <= u < img_viz.shape[1] and 0 <= v < img_viz.shape[0]:
                    cv2.circle(img_viz, (u, v), radius, color, -1)
    return img_viz

if __name__ == "__main__":
    process_video(
        "/home/kchalabi/Documents/StraightWalking/camera_0.mp4",
        "/home/kchalabi/Documents/yolo11m.pt",
        "/home/kchalabi/Documents/weights/nlf/nlf_s_multi_0.2.2.torchscript"
    )
    
