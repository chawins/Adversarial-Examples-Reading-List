# Adversarial-Examples-Reading-List
There are many publications in this area coming out almost every week so I want to keep track of the ones I have read and the ones I plan to read, in case it will be useful to anyone who is also interested in the concept of adversarial examples.

## Background
- C. Szegedy, J. Bruna, D. Erhan, and I. Goodfellow, “Intriguing properties of neural networks.”
- I. J. Goodfellow, J. Shlens, and C. Szegedy, “EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES.”
- A. Nguyen, J. Yosinski, and J. Clune, “Deep Neural Networks are Easily Fooled,” CVPR, 2015 IEEE Conf., pp. 427–436, 2015.
- N. Papernot, P. Mcdaniel, I. Goodfellow, S. Jha, Z. B. Celik, and A. Swami, “Practical Black-Box Attacks against Deep Learning Systems using Adversarial Examples.”
- N. Papernot, P. McDaniel, S. Jha, M. Fredrikson, Z. B. Celik, and A. Swami, “The Limitations of Deep Learning in Adversarial Settings,” IEEE, Nov. 2015.
- N. Papernot, P. Mcdaniel, A. Sinha, and M. Wellman, “SoK : Towards the Science of Security and Privacy in Machine Learning.”

## Attacks
- N. Carlini and D. Wagner, “Towards Evaluating the Robustness of Neural Networks.”
- P.-Y. Chen, Y. Sharma, H. Zhang, J. Yi, and C.-J. Hsieh, “EAD: Elastic-Net Attacks to Deep Neural Networks via Adversarial Examples.”
- O. Poursaeed, I. Katsman, B. Gao, and S. Belongie, “Generative Adversarial Perturbations.”
- S. Baluja and I. Fischer, “Adversarial Transformation Networks: Learning to Generate Adversarial Examples.”
  - Train neural net to generate adversarial examples
- F. Tramèr, A. Kurakin, N. Papernot, D. Boneh, and P. Mcdaniel, “Ensemble Adversarial Training: Attacks and Defenses.”
- Y. Liu, X. Chen, C. Liu, and D. Song, “Delving into Transferable Adversarial Examples and Black-box Attacks,” no. 2, pp. 1–24, 2016.
- S.-M. Moosavi-Dezfooli, A. Fawzi, O. Fawzi, and P. Frossard, “Universal adversarial perturbations,” 2016.
- S.-M. Moosavi-Dezfooli, A. Fawzi, and P. Frossard, “DeepFool: a simple and accurate method to fool deep neural networks,” CVPR, pp. 2574–2582, 2016.
- F. Tramèr, N. Papernot, I. Goodfellow, D. Boneh, and P. Mcdaniel, “The Space of Transferable Adversarial Examples.”
- M. Cisse, Y. Adi, N. Neverova, and J. Keshet, “Houdini: Fooling Deep Structured Prediction Models.”
  - Generating adversarial examples using surrogate loss in place of real non-differentiable task loss

## Defenses
- N. Papernot, P. McDaniel, X. Wu, S. Jha, and A. Swami, “Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks,” 2015.
- A. Kurakin, G. Brain, I. J. Goodfellow, and S. Bengio, “ADVERSARIAL MACHINE LEARNING AT SCALE.”
  - Adversarial training
- D. Meng and H. Chen, “MagNet: a Two-Pronged Defense against Adversarial Examples.”
- S. Shen, G. Jin, K. Gao, and Y. Zhang, “APE-GAN: Adversarial Perturbation Elimination with GAN.”
- A. Mądry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, “Towards Deep Learning Models Resistant to Adversarial Attacks.”
- S. Zheng, T. Leung, and I. Goodfellow, “Improving the Robustness of Deep Neural Networks via Stability Training.”

## Robustness Analysis
- O. Bastani, Y. Ioannou, L. Lampropoulos, D. Vytiniotis, A. V Nori, and A. Criminisi, “Measuring Neural Net Robustness with Constraints.”
- A. Fawzi, O. Fawzi, and P. Frossard, “Analysis of classifiers’ robustness to adversarial perturbations.”
- A. Matyasko and L. Chau, “Margin Maximization for Robust Classification using Deep Learning,” pp. 300–307, 2017.

## Applications
- Physically robust adversarial examples
  - A. Kurakin, I. J. Goodfellow, and S. Bengio, “ADVERSARIAL EXAMPLES IN THE PHYSICAL WORLD.”
  - A. Athalye, L. Engstrom, A. Ilyas, and K. Kwok, “SYNTHESIZING ROBUST ADVERSARIAL EXAMPLES.”
  - J. Lu, H. Sibai, E. Fabry, and D. Forsyth, “NO Need to Worry about Adversarial Examples in Object Detection in Autonomous Vehicles,” 2017.
  - J. Lu, H. Sibai, E. Fabry, and D. Forsyth, “Standard detectors aren’t (currently) fooled by physical adversarial stop signs.”
- C. Xie, J. Wang, Z. Zhang, Y. Zhou, L. Xie, and A. Yuille, “Adversarial Examples for Semantic Segmentation and Object Detection.”
- N. Carlini et al., “Hidden Voice Commands,” USENIX, 2016.
- S. Huang, N. Papernot, I. Goodfellow, Y. Duan, and P. Abbeel, “Adversarial Attacks on Neural Network Policies.”
- J. Lu, H. Sibai, and E. Fabry, “Adversarial Examples that Fool Detectors.”

## Etc.
- Neural network verification
  - X. Huang, M. Kwiatkowska, S. Wang, and M. Wu, “Safety Verification of Deep Neural Networks.”
  - G. Katz, C. Barrett, D. Dill, K. Julian, and M. Kochenderfer, “Reluplex: An Efficient SMT Solver for Verifying Deep Neural Networks.”
  - N. Carlini, G. Katz, C. Barrett, and D. L. Dill, “Ground-Truth Adversarial Examples.”
- Defeating defenses
  - N. Carlini and D. Wagner, “Adversarial Examples Are Not Easily Detected: Bypassing Ten Detection Methods.”
  - W. He, J. Wei, X. Chen, and N. Carlini, “Adversarial Example Defenses: Ensembles of Weak Defenses are not Strong.”
  
# Useful Links
- https://github.com/tensorflow/cleverhans

## To-Read
- W. Brendel, J. Rauber, and M. Bethge, “DECISION-BASED ADVERSARIAL ATTACKS: RELIABLE ATTACKS AGAINST BLACK-BOX MACHINE LEARNING MODELS,” 2017.
- A. Bagnall, R. Bunescu, and G. Stewart, “Training Ensembles to Detect Adversarial Examples,” 2017.
- K. Grosse, D. Pfaff, M. T. Smith, and M. Backes, “How Wrong Am I? — Studying Adversarial Examples and their Impact on Uncertainty in Gaussian Process Machine Learning Models.”
- E. D. Cubuk, B. Zoph, S. S. Schoenholz, Q. V Le, and G. Brain, “INTRIGUING PROPERTIES OF ADVERSARIAL EXAMPLES.”
- Z. Sun, M. Ozay, and T. Okatani, “HYPERNETWORKS WITH STATISTICAL FILTERING FOR DEFENDING ADVERSARIAL EXAMPLES.”
- C. Xie, Z. Zhang, A. L. Yuille, J. Wang, and Z. Ren, “MITIGATING ADVERSARIAL EFFECTS THROUGH RANDOMIZATION.”
- K. Grosse, P. Manoharan, N. Papernot, M. Backes, and P. Mcdaniel, “On the (Statistical) Detection of Adversarial Examples.”
- J. Z. Kolter and E. Wong, “Provable defenses against adversarial examples via the convex outer adversarial polytope,” 2017.
- Z. Zhao, D. Dua, and S. Singh, “GENERATING NATURAL ADVERSARIAL EXAMPLES.”
- Y. Song, T. Kim, S. Nowozin, S. Ermon, and N. Kushman, “PIXELDEFEND: LEVERAGING GENERATIVE MODELS TO UNDERSTAND AND DEFEND AGAINST ADVERSARIAL EXAMPLES.”
- J. Sokolic, R. Giryes, G. Sapiro, and M. R. D. Rodrigues, “Robust Large Margin Deep Neural Networks,” May 2016.
- R. Huang, B. Xu, D. Schuurmans, and C. Szepesvári, “LEARNING WITH A STRONG ADVERSARY.”
- P.-Y. Chen, H. Zhang, Y. Sharma, J. Yi, and C.-J. Hsieh, “ZOO: Zeroth Order Optimization based Black-box Attacks to Deep Neural Networks without Training Substitute Models.”
- J. Hayes and G. Danezis, “Machine Learning as an Adversarial Service: Learning Black-Box Adversarial Examples.”
- X. Cao and N. Z. Gong, “Mitigating Evasion Attacks to Deep Neural Networks via Region-based Classification.”
- N. Narodytska and S. Kasiviswanathan, “Simple Black-Box Adversarial Attacks on Deep Neural Networks.”
- H. Hosseini, Y. Chen, S. Kannan, B. Zhang, and R. Poovendran, “Blocking Transferability of Adversarial Examples in Black-Box Learning Systems.”
- T. Pang, C. Du, Y. Dong, and J. Zhu, “Towards Robust Detection of Adversarial Examples.” [link](https://arxiv.org/pdf/1706.00633.pdf)
- X. Yuan, P. He, Q. Zhu, R. R. Bhat, and X. Li, “Adversarial Examples: Attacks and Defenses for Deep Learning.” [link](https://arxiv.org/pdf/1712.07107.pdf)
- A. Ilyas, L. Engstrom, A. Athalye, and J. Lin, “Query-efficient Black-box Adversarial Examples.” [link](https://arxiv.org/pdf/1712.07113.pdf)
- A. Rawat, M. Wistuba, and M.-I. Nicolae, “Harnessing Model Uncertainty for Detecting Adversarial Examples.” [link](http://bayesiandeeplearning.org/2017/papers/37.pdf)
- J. Ebrahimi, A. Rao, D. Lowd, and D. Dou, “HotFlip: White-Box Adversarial Examples for NLP.”
- F. Suya, Y. Tian, D. Evans, and P. Papotti, “Query-limited Black-box Attacks to Classifiers.”
