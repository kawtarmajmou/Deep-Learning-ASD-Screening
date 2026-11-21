"""
ASD Detection Web App
=====================
Run with:  python webapp/app.py
Then open: http://localhost:8080
"""

import os, sys, json, io, pickle
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.impute import SimpleImputer
from scipy.special import expit, logit

# ── CONFIG ────────────────────────────────────────────────────────────────────
EEG_MODEL_PATH = ROOT / 'results' / 'lstm_v2_best.pth'
MRI_MODEL_PATH = ROOT / 'results' / 'mri_model.pkl'
DEVICE         = torch.device('cpu')
SEQ_LEN        = 5


# ── EEG MODEL ─────────────────────────────────────────────────────────────────
class LightCNNLSTM(nn.Module):
    def __init__(self, sl=5):
        super().__init__()
        self.sl  = sl
        self.cnn = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4,4)))
        self.lstm = nn.LSTM(1024, 256, num_layers=2, batch_first=True, dropout=0.4)
        self.fc   = nn.Sequential(
            nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128,2))

    def forward(self, x):
        B, T, C, H, W = x.shape
        feats = [self.cnn(x[:,t]).view(B,-1) for t in range(T)]
        _, (hn,_) = self.lstm(torch.stack(feats, 1))
        return self.fc(hn[-1])

TRANSFORM = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("Loading EEG model ...", flush=True)
eeg_model = LightCNNLSTM(sl=SEQ_LEN)
eeg_model.load_state_dict(torch.load(EEG_MODEL_PATH, map_location=DEVICE))
eeg_model.eval()
print(f"EEG model loaded.", flush=True)

print("Loading MRI model ...", flush=True)
with open(MRI_MODEL_PATH, 'rb') as f:
    mri_data = pickle.load(f)
mri_model     = mri_data['model']
mri_type      = mri_data.get('type', 'connectivity')
mri_threshold = mri_data.get('threshold', 0.5)
mri_n_sites   = mri_data.get('n_sites', 20)
print(f"MRI model loaded (type={mri_type}).", flush=True)


# ── EEG PREDICTION ────────────────────────────────────────────────────────────
def predict_eeg(image_bytes_list):
    if len(image_bytes_list) == 0:
        return {"error": "No images uploaded."}
    if len(image_bytes_list) < SEQ_LEN:
        return {"error": f"Need at least {SEQ_LEN} spectrogram images. Got {len(image_bytes_list)}."}

    image_bytes_list.sort(key=lambda x: x[0])

    frames = []
    for fname, data in image_bytes_list:
        try:
            img = Image.open(io.BytesIO(data)).convert('RGB')
            frames.append(TRANSFORM(img))
        except Exception as e:
            return {"error": f"Failed to read image {fname}: {e}"}

    sequences = []
    for i in range(0, len(frames) - SEQ_LEN + 1, 2):
        seq = torch.stack(frames[i:i+SEQ_LEN])
        sequences.append(seq)

    if not sequences:
        return {"error": "Could not build sequences from uploaded images."}

    autism_prob_sum = 0.0
    with torch.no_grad():
        for seq in sequences:
            x      = seq.unsqueeze(0)
            logits = eeg_model(x)
            prob   = F.softmax(logits, dim=1)[0, 1].item()
            autism_prob_sum += prob

    avg = autism_prob_sum / len(sequences)
    prediction = "Autism Spectrum" if avg >= 0.5 else "Normal"
    confidence = avg * 100 if avg >= 0.5 else (1 - avg) * 100

    return {
        "prediction":     prediction,
        "confidence":     round(confidence, 1),
        "autism_prob":    round(avg * 100, 1),
        "normal_prob":    round((1 - avg) * 100, 1),
        "total_segments": len(image_bytes_list),
        "sequences_used": len(sequences),
    }


# ── MRI PREDICTION ────────────────────────────────────────────────────────────
def parse_1d(data_bytes):
    """Parse a .1D file (bytes) → numpy array (timepoints x ROIs)."""
    rows = []
    for line in data_bytes.decode('utf-8', errors='ignore').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        try:
            rows.append([float(v) for v in line.split()])
        except:
            continue
    return np.array(rows) if rows else None

def fisher_z(r):
    r = np.clip(r, -0.9999, 0.9999)
    return 0.5 * np.log((1 + r) / (1 - r))

def partial_corr_from_corr(corr, alpha=0.1):
    """Regularized partial correlation (direct brain connections)."""
    reg = corr + alpha * np.eye(corr.shape[0])
    try:
        prec = np.linalg.inv(reg)
    except np.linalg.LinAlgError:
        prec = np.linalg.pinv(reg)
    diag = np.sqrt(np.abs(np.diag(prec)))
    diag = np.where(diag < 1e-8, 1e-8, diag)
    pc = -prec / np.outer(diag, diag)
    np.fill_diagonal(pc, 1.0)
    return np.clip(pc, -1.0, 1.0)

def compute_graph_features(corr):
    """Degree + variance + top-10 mean per ROI. (600,)"""
    A = np.abs(corr)
    np.fill_diagonal(A, 0)
    degree   = A.sum(axis=1)
    conn_var = A.std(axis=1)
    top10    = np.sort(A, axis=1)[:, -10:].mean(axis=1)
    return np.concatenate([degree, conn_var, top10]).astype(np.float32)

def build_mri_features(corr, model_type, n_sites):
    """Build the feature vector matching what each model type was trained on."""
    idx = np.triu_indices(200, k=1)

    if model_type == 'combat':
        # Best model: Pearson (Fisher-z) + Partial correlation concatenated
        feat_type = mri_data.get('feature_type', 'pearson')
        z_pear = fisher_z(corr)[idx].astype(np.float32)
        if 'partial' in feat_type or feat_type == 'pearson':
            # Pearson+Partial combination (39800 features)
            pc     = partial_corr_from_corr(corr, alpha=0.1)
            z_part = fisher_z(pc)[idx].astype(np.float32)
            return np.concatenate([z_pear, z_part]).reshape(1, -1)
        elif feat_type == 'dynamic':
            # Dynamic FC not computable from single correlation matrix
            # Fall back to Pearson only
            return z_pear.reshape(1, -1)
        else:
            return z_pear.reshape(1, -1)

    elif model_type == 'ultra':
        z_feat   = fisher_z(corr)[idx].astype(np.float32)
        g_feat   = compute_graph_features(corr)
        clin     = np.full(4, np.nan, dtype=np.float32)
        site_oh  = np.zeros(n_sites, dtype=np.float32)
        return np.concatenate([z_feat, g_feat, clin, site_oh]).reshape(1, -1)

    elif model_type == 'advanced':
        z_feat = fisher_z(corr)[idx].astype(np.float32)
        if mri_data.get('use_graph', False):
            g_feat = compute_graph_features(corr)
            return np.concatenate([z_feat, g_feat]).reshape(1, -1)
        return z_feat.reshape(1, -1)

    else:
        # Default: raw Pearson upper triangle
        return corr[idx].astype(np.float32).reshape(1, -1)

def predict_mri(form_fields):
    """Accept a .1D file upload and predict ASD from functional connectivity."""
    file_data = form_fields.get('mri_file', [None])[0]

    if file_data is None or len(file_data) < 10:
        return {"error": "Please upload a valid .1D file."}

    ts = parse_1d(file_data)
    if ts is None or ts.shape[0] < 10:
        return {"error": "Could not read .1D file. Make sure it is a valid ABIDE ROI time series."}

    # Fix transposed files
    if ts.shape[0] == 200 and ts.shape[1] > 200:
        ts = ts.T

    if ts.shape[1] != 200:
        return {"error": f"Expected 200 brain regions (cc200 atlas), got {ts.shape[1]}."}

    # Compute correlation matrix
    corr = np.corrcoef(ts.T)
    corr = np.nan_to_num(corr, nan=0.0)

    # Build features for this model type
    fc = build_mri_features(corr, mri_type, mri_n_sites)

    # Predict
    prob_arr  = mri_model.predict_proba(fc)[0]
    autism_p  = float(prob_arr[1])
    threshold = mri_threshold
    prediction = "Autism Spectrum" if autism_p >= threshold else "Normal"
    confidence = autism_p * 100 if autism_p >= threshold else (1 - autism_p) * 100

    model_label = {
        'combat':           'SVM + ComBat + Pearson+Partial (ABIDE, AUC=76.05%)',
        'ultra':            'Ensemble (LR+SVM+GBM+MLP, ABIDE 996 subjects)',
        'advanced':         'Advanced Pipeline (Fisher-z + Graph, ABIDE)',
        'feature_selection':'SVM + Feature Selection (ABIDE)',
        'connectivity':     'Logistic Regression (ABIDE fMRI, 1025 subjects)',
        'cnn_ensemble':     'CNN + LR Ensemble (ABIDE)',
    }.get(mri_type, 'MRI Model (ABIDE)')

    return {
        "prediction":    prediction,
        "confidence":    round(confidence, 1),
        "autism_prob":   round(autism_p * 100, 1),
        "normal_prob":   round((1 - autism_p) * 100, 1),
        "model":         model_label,
        "timepoints":    int(ts.shape[0]),
        "brain_regions": int(ts.shape[1]),
    }


# ── MULTIMODAL FUSION ─────────────────────────────────────────────────────────
# Best fusion: Temperature-calibrated ensemble (equal weights)
# T_MRI=1.73, T_EEG=1.02  →  AUC=83.56%, F1=76.94% (simulation-based analysis)
FUSION_T_MRI = 1.73
FUSION_T_EEG = 1.02

def _temp_calibrate(p, T):
    """Apply temperature scaling to a probability."""
    return float(expit(logit(np.clip(p, 0.01, 0.99)) / T))

def predict_multimodal(image_bytes_list, form_fields):
    """Combine EEG spectrograms + MRI .1D file for fusion prediction."""
    # --- EEG arm ---
    eeg_result = predict_eeg(image_bytes_list)
    if "error" in eeg_result:
        return {"error": f"EEG: {eeg_result['error']}"}
    p_eeg_raw = eeg_result["autism_prob"] / 100.0

    # --- MRI arm ---
    mri_result = predict_mri(form_fields)
    if "error" in mri_result:
        return {"error": f"MRI: {mri_result['error']}"}
    p_mri_raw = mri_result["autism_prob"] / 100.0

    # --- Temperature calibration + equal-weight average ---
    pc_mri  = _temp_calibrate(p_mri_raw, FUSION_T_MRI)
    pc_eeg  = _temp_calibrate(p_eeg_raw, FUSION_T_EEG)
    p_fused = 0.5 * pc_mri + 0.5 * pc_eeg

    prediction = "Autism Spectrum" if p_fused >= 0.5 else "Normal"
    confidence = p_fused * 100 if p_fused >= 0.5 else (1 - p_fused) * 100

    return {
        "prediction":      prediction,
        "confidence":      round(confidence, 1),
        "autism_prob":     round(p_fused * 100, 1),
        "normal_prob":     round((1 - p_fused) * 100, 1),
        "eeg_prob":        round(p_eeg_raw * 100, 1),
        "mri_prob":        round(p_mri_raw * 100, 1),
        "fusion_method":   "Temperature-calibrated ensemble (T_MRI=1.73, T_EEG=1.02, w=0.50/0.50)",
        "model":           "Multimodal EEG+fMRI (AUC=83.56%, F1=76.94%)",
        "eeg_sequences":   eeg_result.get("sequences_used", 0),
        "mri_timepoints":  mri_result.get("timepoints", 0),
        "mri_regions":     mri_result.get("brain_regions", 0),
    }


# ── HTTP HANDLER ──────────────────────────────────────────────────────────────
HTML_PATH = Path(__file__).parent / 'index.html'

class Handler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        print(f"  {self.address_string()} - {format % args}", flush=True)

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            html = HTML_PATH.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', len(html))
            self.end_headers()
            self.wfile.write(html)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == '/predict_eeg':
            self._handle_eeg()
        elif self.path == '/predict_mri':
            self._handle_mri()
        elif self.path == '/predict_multimodal':
            self._handle_multimodal()
        else:
            self.send_error(404)

    def _parse_multipart(self):
        """Parse multipart form data from request"""
        content_type = self.headers.get('Content-Type', '')
        if not content_type.startswith('multipart/form-data'):
            return None

        try:
            # Extract boundary
            boundary_str = content_type.split('boundary=')[1]
            if ';' in boundary_str:
                boundary_str = boundary_str.split(';')[0]
            boundary = boundary_str.strip().strip('"').encode()

            # Read body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)

            # Parse parts
            fields = {}
            parts = body.split(b'--' + boundary)

            for part in parts[1:-1]:  # Skip first boundary and last closing boundary
                if not part.strip():
                    continue

                # Split headers from content
                if b'\r\n\r\n' in part:
                    header_section, content = part.split(b'\r\n\r\n', 1)
                else:
                    continue

                # Parse Content-Disposition to get field name
                header_str = header_section.decode('utf-8', errors='ignore')
                if 'name=' not in header_str:
                    continue

                # Extract field name
                name_start = header_str.find('name="') + 6
                name_end = header_str.find('"', name_start)
                field_name = header_str[name_start:name_end]

                # Remove trailing CRLF from content
                if content.endswith(b'\r\n'):
                    content = content[:-2]

                # Add to fields
                if field_name not in fields:
                    fields[field_name] = []
                fields[field_name].append(content)

            return fields if fields else None
        except Exception as e:
            print(f"Multipart parsing error: {e}", flush=True)
            return None

    def _handle_eeg(self):
        fields = self._parse_multipart()
        if fields is None:
            self._json(400, {"error": "Expected multipart/form-data"})
            return

        images    = fields.get('images[]', []) or fields.get('images', [])
        filenames = fields.get('filenames[]', []) or fields.get('filenames', [])

        named = []
        for i, data in enumerate(images):
            fname = filenames[i].decode() if i < len(filenames) and isinstance(filenames[i], bytes) \
                    else filenames[i] if i < len(filenames) else f"seg_{i:04d}.png"
            named.append((fname, data))

        self._json(200, predict_eeg(named))

    def _handle_mri(self):
        fields = self._parse_multipart()
        if fields is None:
            self._json(400, {"error": "Expected multipart/form-data"})
            return
        self._json(200, predict_mri(fields))

    def _handle_multimodal(self):
        fields = self._parse_multipart()
        if fields is None:
            self._json(400, {"error": "Expected multipart/form-data"})
            return

        images    = fields.get('images[]', []) or fields.get('images', [])
        filenames = fields.get('filenames[]', []) or fields.get('filenames', [])
        named = []
        for i, data in enumerate(images):
            fname = filenames[i].decode() if i < len(filenames) and isinstance(filenames[i], bytes) \
                    else filenames[i] if i < len(filenames) else f"seg_{i:04d}.png"
            named.append((fname, data))

        self._json(200, predict_multimodal(named, fields))

    def _json(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port   = 8080
    server = HTTPServer(('0.0.0.0', port), Handler)
    print(f"\n{'='*50}")
    print(f"  ASD Detection Web App")
    print(f"  Open in browser: http://localhost:{port}")
    print(f"{'='*50}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
