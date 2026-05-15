# =============================================================
# Element Quest — Training Configuration
# Edit this file to control training behaviour.
# No restart needed — changes apply at the next training run.
# =============================================================

# Set True  → a Pygame window opens during training so you can
#             watch the agent learn every episode in real time.
#             Training is somewhat slower (display overhead).
# Set False → headless training (fastest). A small preview
#             window pops up every RENDER_PREVIEW_EVERY episodes.
RENDER_TRAINING = False

# When RENDER_TRAINING = False, a preview window appears every
# N training episodes. Set to 0 to disable previews entirely.
RENDER_PREVIEW_EVERY = 0

# Max frames per second for the training render window.
# 0 = uncapped (absolute fastest).
# 300 is a good balance — smooth enough to watch, fast to train.
RENDER_FPS = 300
