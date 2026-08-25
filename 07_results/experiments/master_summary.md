# Master Summary of All Experiments

| Experiment                    | Dataset                         | Model               |   Accuracy |   F1_Macro |
|:------------------------------|:--------------------------------|:--------------------|-----------:|-----------:|
| EXP-01 (CNN Baseline)         | project_design                  | EfficientNet-B0     |     0.8837 |     0.8635 |
| EXP-01 (CNN Baseline)         | sitting_posture_detection       | EfficientNet-B0     |     0.8493 |     0.841  |
| EXP-02 (Keypoint Classifiers) | Postureexercise (7-KP, 5-class) | MLP                 |     0.8734 |     0.8419 |
| EXP-02 (Keypoint Classifiers) | Postureexercise (7-KP, 5-class) | XGBoost             |     0.8481 |     0.8155 |
| EXP-02 (Keypoint Classifiers) | IKORN (4-KP, 2-class)           | MLP                 |     0.9293 |     0.9051 |
| EXP-02 (Keypoint Classifiers) | IKORN (4-KP, 2-class)           | XGBoost             |     0.9697 |     0.9569 |
| EXP-03 (YOLO Pose + Clf)      | project_design                  | YOLO-Pose + MLP     |     0.8372 |     0.8207 |
| EXP-03 (YOLO Pose + Clf)      | project_design                  | YOLO-Pose + XGBoost |     0.7984 |     0.7855 |
| EXP-03 (YOLO Pose + Clf)      | sitting_posture_detection       | YOLO-Pose + MLP     |     0.6027 |     0.4925 |
| EXP-03 (YOLO Pose + Clf)      | sitting_posture_detection       | YOLO-Pose + XGBoost |     0.8219 |     0.7745 |
