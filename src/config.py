"""All hyperparameters in one place. To switch from prototype to full
implementation, change USE_REAL_MODELS = True and provide HF model paths."""

# ----- Run configuration -----
SEED = 42
USE_REAL_MODELS = False
EMBED_DIM = 128          # toy encoder; for BERT this is 768

# ----- Data -----
DATA_SCALE = 0.4         # 0.4 = ~1500 posts; 1.0 = full Sha-Forum size (~3700)
TARGET_BALANCE = 1413    # majority cell size at scale=1.0; auto-rescaled in code

# ----- Step A: IH_class -----
N_CV_FOLDS = 5
HARDNESS_LEARNERS = ["lr", "svm_rbf", "rf"]

# ----- Step B1: vocab debiasing -----
TFIDF_MAX_FEATURES = 3000
TFIDF_NGRAM = (1, 2)
LR_C_L1 = 4.0           # higher C = less regularization, richer V_disc
BONFERRONI_ALPHA = 0.05
TOP_K_DISC_VOCAB = 80   # need more bans to neutralize both groups

# ----- Step B2: LLM generation -----
GEN_TEMPERATURE = 0.85
GEN_OVERSAMPLE_FACTOR = 2.5

# ----- Step C1: label fidelity -----
TAU_LABEL = 0.65

# ----- Step C2: probe -----
N_PROBES = 5
PROBE_HIDDEN = 64
PROBE_EPOCHS = 200
PROBE_LR = 1e-3
PROBE_DA_WEIGHT = 0.3
PROBE_VARIANCE_TAU = 0.05
PROBE_MEAN_LOW = 0.30
PROBE_MEAN_HIGH = 0.70
PROBE_ENTROPY_MIN = 0.65

# ----- Step C3: diversity -----
DIVERSITY_DELTA = 1.0

# ----- Step C4: MMD -----
MMD_P_VALUE_MIN = 0.05
MMD_PERMUTATIONS = 200

# ----- Step D: Method-3 -----
METHOD3_K = 25           # number of resampling iterations

# ----- Step E -----
ABROCA_GRID = 200

# ----- Experiment -----
N_SEEDS = 10             # number of seeds for the experiment
