# MonoTher-Depth: Enhancing Thermal Depth Estimation via Confidence-Aware Distillation

The code will be released soon!

This repository contains code necessary to run our Monocular depth estimation (MDE)  from thermal images. 


## Abstract
Monocular depth estimation (MDE) from thermal images is a crucial technology for robotic systems operating in challenging conditions such as fog, smoke, and low light. The limited availability of labeled thermal data constrains the generalization capabilities of thermal MDE models compared to foundational RGB MDE models, which benefit from datasets of millions of images across diverse scenarios. To address this challenge, we introduce a novel pipeline that enhances thermal MDE through knowledge distillation from a versatile RGB MDE model. Our approach features a confidence-aware distillation method that utilizes the predicted confidence of the RGB MDE to selectively strengthen the thermal MDE model, capitalizing on the strengths of the RGB model while mitigating its weaknesses. Our method significantly improves the accuracy of the thermal MDE, independent of the availability of labeled depth supervision, and greatly expands its applicability to new scenarios.
In our experiments on new scenarios without labeled depth, the proposed confidence-aware distillation method reduces the absolute relative error of thermal MDE by 22.88\% compared to the baseline without distillation.








## Citation
```
@article{zuo2025monotherdepth,
  author={Zuo, Xingxing and Ranganathan, Nikhil and Lee, Connor and Gkioxari, Georgia and Chung, Soon-Jo},
  journal={IEEE Robotics and Automation Letters}, 
  title={MonoTher-Depth: Enhancing Thermal Depth Estimation via Confidence-Aware Distillation}, 
  year={2025}
}
```

