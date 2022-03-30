IGNORE_INDEX = -100
NON_LABEL = -1
classes = ['Lead', 'Position', 'Claim','Counterclaim', 'Rebuttal','Evidence','Concluding Statement']
OUTPUT_LABELS = ['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim', 'I-Counterclaim', 
                 'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement', 'I-Concluding Statement']
LABELS_TO_IDS = {v:k for k,v in enumerate(OUTPUT_LABELS)}
IDS_TO_LABELS = {k:v for k,v in enumerate(OUTPUT_LABELS)}
MIN_THRESH = {
    "I-Lead": 9,
    "I-Position": 5,
    "I-Evidence": 14,
    "I-Claim": 3,
    "I-Concluding Statement": 11,
    "I-Counterclaim": 6,
    "I-Rebuttal": 4,
}
# MIN_THRESH = {
#     "I-Lead": 0,
#     "I-Position": 0,
#     "I-Evidence": 0,
#     "I-Claim": 0,
#     "I-Concluding Statement": 0,
#     "I-Counterclaim": 0,
#     "I-Rebuttal": 0,
# }
# MIN_THRESH = {
#     "I-Lead": 0,
#     "I-Position": 0,
#     "I-Evidence": 0,
#     "I-Claim": 0,
#     "I-Concluding Statement": 0,
#     "I-Counterclaim": 0,
#     "I-Rebuttal": 0,
# }

PROB_THRESH = {
    "I-Lead": 0.7,
    "I-Position": 0.55,
    "I-Evidence": 0.65,
    "I-Claim": 0.55,
    "I-Concluding Statement": 0.7,
    "I-Counterclaim": 0.5,
    "I-Rebuttal": 0.55,
}
# PROB_THRESH = {
#     "I-Lead": 0.0,
#     "I-Position": 0.0,
#     "I-Evidence": 0.0,
#     "I-Claim": 0.0,
#     "I-Concluding Statement": 0.0,
#     "I-Counterclaim": 0.0,
#     "I-Rebuttal": 0.0
# }