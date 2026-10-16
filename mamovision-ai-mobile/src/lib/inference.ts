/**
 * Inference entry point for MobileFCMViTv3.
 *
 * Loads assets/models/mobilefcmvitv3.onnx via onnxruntime-react-native.
 * Model input  : "image"         float32  [1, 3, 224, 224]
 * Model output : "probabilities" float32  [1, 3]
 * Class order  : [benign=0, malignant=1, normal=2]  (sorted BUSI folder names)
 */

import * as ort from 'onnxruntime-react-native';
import { Asset } from 'expo-asset';
import { AnalysisResult, BiRadsClass } from '../types';
import { preprocessForModel, ModelInputs } from './preprocessing';

// ─── BI-RADS display content ──────────────────────────────────────────────────
const OBSERVATIONS: Record<BiRadsClass, string[]> = {
  'BI-RADS 1': [
    'No suspicious findings detected',
    'Normal breast tissue architecture',
    'No mass, distortion, or calcification',
    'Negative study',
  ],
  'BI-RADS 2': [
    'Benign finding present',
    'Simple cyst identified',
    'No malignant features observed',
    'Routine follow-up recommended',
  ],
  'BI-RADS 3': [
    'Probably benign finding',
    'Short-interval follow-up suggested',
    'Low suspicion for malignancy',
    'Six-month follow-up recommended',
  ],
  'BI-RADS 4': [
    'Suspicious abnormality detected',
    'Biopsy should be considered',
    'Irregular hypoechoic mass noted',
    'Moderate to high suspicion for malignancy',
  ],
  'BI-RADS 5': [
    'Highly suspicious for malignancy',
    'Biopsy strongly recommended',
    'Spiculated mass with posterior shadowing',
    'High probability of malignancy',
  ],
};

const ANALYSIS_TEXTS: Record<BiRadsClass, string> = {
  'BI-RADS 1':
    'NEGATIVE\n\nThe ultrasound examination demonstrates normal breast parenchyma without evidence of suspicious masses, architectural distortion, or abnormal calcifications. Breast tissue appears homogeneous with normal ductal architecture.\n\nRecommendation: Routine annual screening as per age-appropriate guidelines.',
  'BI-RADS 2':
    'BENIGN\n\nA benign finding is identified. The lesion demonstrates classic benign features including well-circumscribed margins, anechoic interior, and posterior acoustic enhancement consistent with a simple cyst.\n\nRecommendation: Routine follow-up. No additional imaging or intervention required.',
  'BI-RADS 3':
    'PROBABLY BENIGN\n\nA probably benign finding is present with less than 2% risk of malignancy based on imaging characteristics. The lesion demonstrates predominantly benign features.\n\nRecommendation: Short-interval follow-up ultrasound in 6 months to assess stability.',
  'BI-RADS 4':
    'SUSPICIOUS\n\nA suspicious lesion requiring tissue sampling is identified. The mass demonstrates irregular margins, heterogeneous echotexture, and shadowing raising concern for malignancy.\n\nRecommendation: Ultrasound-guided core needle biopsy is recommended for histological diagnosis.',
  'BI-RADS 5':
    'HIGHLY SUGGESTIVE OF MALIGNANCY\n\nHighly suspicious findings are present. A spiculated hypoechoic mass with posterior acoustic shadowing, taller-than-wide orientation, and angular margins is identified, consistent with invasive malignancy.\n\nRecommendation: Tissue biopsy required. Multidisciplinary oncology consultation recommended.',
};

// ─── Metrics ──────────────────────────────────────────────────────────────────
export interface InferenceMetrics {
  modelStatus: 'loading' | 'ready' | 'error';
  modelSizeBytes: number | null;
  lastInferenceMs: number | null;
  avgInferenceMs: number | null;
  totalInferences: number;
  lastInferenceAt: string | null;
}

const metrics: InferenceMetrics = {
  modelStatus: 'loading',
  modelSizeBytes: null,
  lastInferenceMs: null,
  avgInferenceMs: null,
  totalInferences: 0,
  lastInferenceAt: null,
};

export function getInferenceMetrics(): InferenceMetrics { return { ...metrics }; }

// ─── ONNX session (singleton) ─────────────────────────────────────────────────
// Class indices as saved by BUSIDataset (sorted BUSI folder names: benign, malignant, normal)
const IDX_BENIGN    = 0;
const IDX_MALIGNANT = 1;
const IDX_NORMAL    = 2;

let _session: ort.InferenceSession | null = null;
let _loadPromise: Promise<ort.InferenceSession> | null = null;

async function loadModel(): Promise<ort.InferenceSession> {
  if (_session) return _session;
  if (_loadPromise) return _loadPromise;

  _loadPromise = (async () => {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const asset = Asset.fromModule(require('../../assets/models/mobilefcmvitv3.onnx'));
    await asset.downloadAsync();

    const sess = await ort.InferenceSession.create(asset.localUri!, {
      executionProviders: ['cpu'],
    });

    _session = sess;
    metrics.modelStatus = 'ready';
    metrics.modelSizeBytes = asset.fileSize ?? null;
    return sess;
  })();

  return _loadPromise;
}

// Kick off model loading as soon as this module is imported so the session
// is warm before the user picks an image.
loadModel().catch(() => { metrics.modelStatus = 'error'; });

// ─── ONNX inference ───────────────────────────────────────────────────────────
async function runModel(inputs: ModelInputs): Promise<[number, number, number]> {
  const sess = await loadModel();

  const tensor = new ort.Tensor('float32', inputs.imageTensor, [1, 3, 224, 224]);
  const output = await sess.run({ image: tensor });
  const probs  = output['probabilities'].data as Float32Array;

  // Reorder from training class order [benign, malignant, normal]
  // to the order expected by mapToResult: [normal, benign, malignant]
  return [probs[IDX_NORMAL], probs[IDX_BENIGN], probs[IDX_MALIGNANT]];
}

// ─── BI-RADS mapping ──────────────────────────────────────────────────────────
// Maps the 3-class model output to the BI-RADS scale used in clinical display.
// normal    → BI-RADS 1
// benign    → BI-RADS 2 (high confidence) or 3 (moderate)
// malignant → BI-RADS 4 (< 0.65) or 5 (≥ 0.65)
function mapToResult(probs: [number, number, number]): AnalysisResult {
  const [normal, benign, malignant] = probs;

  let classification: BiRadsClass;
  let confidence: number;

  if (normal >= benign && normal >= malignant) {
    classification = 'BI-RADS 1';
    confidence = normal;
  } else if (benign >= malignant) {
    classification = benign >= 0.55 ? 'BI-RADS 2' : 'BI-RADS 3';
    confidence = benign;
  } else {
    classification = malignant >= 0.65 ? 'BI-RADS 5' : 'BI-RADS 4';
    confidence = malignant;
  }

  return {
    classification,
    confidence,
    observations: OBSERVATIONS[classification],
    analysisText:  ANALYSIS_TEXTS[classification],
  };
}

// ─── Public API ───────────────────────────────────────────────────────────────
export async function analyzeImage(imageUri: string): Promise<AnalysisResult> {
  const start = Date.now();

  const inputs = await preprocessForModel(imageUri);
  const probs  = await runModel(inputs);
  const result = mapToResult(probs);

  const elapsed = Date.now() - start;
  metrics.totalInferences += 1;
  metrics.lastInferenceMs  = elapsed;
  metrics.lastInferenceAt  = new Date().toISOString();
  metrics.avgInferenceMs   =
    metrics.avgInferenceMs === null
      ? elapsed
      : Math.round(
          (metrics.avgInferenceMs * (metrics.totalInferences - 1) + elapsed) /
          metrics.totalInferences,
        );

  return result;
}

export function getClassificationColor(classification?: string): string {
  switch (classification) {
    case 'BI-RADS 1': return '#00FF88';
    case 'BI-RADS 2': return '#00F2FF';
    case 'BI-RADS 3': return '#FFB822';
    case 'BI-RADS 4': return '#FF6B35';
    case 'BI-RADS 5': return '#FF3B3B';
    default:          return '#8B8FA8';
  }
}
