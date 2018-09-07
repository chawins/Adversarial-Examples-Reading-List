# Adversarial Examples Reading List
There are many publications in this area coming out almost every week so I want to keep track of the ones I have read and the ones I plan to read, in case it will be useful to anyone who is also interested in the concept of adversarial examples. The order of the papers is arbitrary. Any paper suggestion is very welcomed.

## Background
- :+1: &nbsp; C. Szegedy, J. Bruna, D. Erhan, and I. Goodfellow, “Intriguing Properties of Neural Networks,” ICLR 2014. [[link]](https://arxiv.org/abs/1312.6199)
- :+1: &nbsp; I. J. Goodfellow, J. Shlens, and C. Szegedy, “Explaining and Harnessing Adversarial Examples,” ICLR 2015. [[link]](https://arxiv.org/abs/1412.6572) 
- A. Nguyen, J. Yosinski, and J. Clune, “Deep Neural Networks are Easily Fooled,” CVPR, 2015 IEEE Conf., pp. 427–436, 2015.
- N. Papernot, P. Mcdaniel, I. Goodfellow, S. Jha, Z. B. Celik, and A. Swami, “Practical Black-Box Attacks against Deep Learning Systems using Adversarial Examples.”
- N. Papernot, P. McDaniel, S. Jha, M. Fredrikson, Z. B. Celik, and A. Swami, “The Limitations of Deep Learning in Adversarial Settings,” IEEE, Nov. 2015.
- N. Papernot, P. Mcdaniel, A. Sinha, and M. Wellman, “SoK : Towards the Science of Security and Privacy in Machine Learning.”
  
---

## Attacks
- :+1: &nbsp; N. Carlini and D. Wagner, “Towards Evaluating the Robustness of Neural Networks.”
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
- W. Brendel, J. Rauber, and M. Bethge, “Decision-Based Adversarial Attacks: Reliable Attacks Against Black-Box Machine Learning Models,” 2017. [[link]](https://arxiv.org/abs/1712.04248)
  - Attack that requires only the classifier's output (# of queries $\approx10^5$). Start with an image of target class and move towards a desired benign image.
- Xiao et al., "SPATIALLY TRANSFORMED ADVERSARIAL EXAMPLES," 2018.
### Attacks with GAN
- Z. Zhao, D. Dua, and S. Singh, “GENERATING NATURAL ADVERSARIAL EXAMPLES.”
- J. Hayes, G. Danezis, "Learning Universal Adversarial Perturbations with Generative Models."
- Xiao et al., "GENERATING ADVERSARIAL EXAMPLES WITH ADVERSARIAL NETWORKS," 2018.
- Poursaeed et al., "Generative Adversarial Perturbations," 2017.
  
---

## Defenses
- N. Papernot, P. McDaniel, X. Wu, S. Jha, and A. Swami, “Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks,” 2015.
- A. Kurakin, G. Brain, I. J. Goodfellow, and S. Bengio, “Adversarial Machine Learning at Scale.”
  - First introduction of adversarial training with FGSM
- S. Gu, L. Rigazio, "Towards Deep Neural Network Architectures Robust to Adversarial Examples," 2015.
- :+1: &nbsp; A. Mądry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, “Towards Deep Learning Models Resistant to Adversarial Attacks.”
  - Adversarial training with PGD provides strong defense (MNIST, CIFAR) even in white-box setting
- S. Zheng, T. Leung, and I. Goodfellow, “Improving the Robustness of Deep Neural Networks via Stability Training.”
- :+1: &nbsp; H. Kannan, A. Kurakin, I. Goodfellow, "Adversarial Logit Pairing," 2018.
- A. Galloway, T. Tanay, G. Taylor, "Adversarial Training Versus Weight Decay," 2018.
- A. Mosca, and G. Magoulas, "Hardening against adversarial examples with the smooth gradient method," 2018.
- A. Raghunathan, J. Steinhardt, and P. Liang, "Certified Defenses against Adversarial Examples," 2018.
- W. Xu, D. Evans, and Q. Yanjun, "Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks," NDSS 2018. [[link]](https://arxiv.org/abs/1704.01155)
  - Experiment with three "feature squeezing": reduce bit depth, local smoothing, non-local smoothing 
  - Evaluated on MNIST, CIFAR-10, ImageNet. Some performance drop on CIFAR-10 and ImageNet
  - Each method works well with different types of norms (i.e. bit depth reduction is very good against $L_2$ or $L_\infty$, smoothing is good against $L_0$, etc.)
  - Can be used as a detector by comparing ($L_1$ distance) logits of the input before and after squeezing
  - Obvious adaptive adversary does not succeed 
### Defenses with GAN, VAE
- Y. Song, T. Kim, S. Nowozin, S. Ermon, and N. Kushman, “PIXELDEFEND: LEVERAGING GENERATIVE MODELS TO UNDERSTAND AND DEFEND AGAINST ADVERSARIAL EXAMPLES.”
- S. Shen, G. Jin, K. Gao, and Y. Zhang, “APE-GAN: Adversarial Perturbation Elimination with GAN.”
- D. Meng and H. Chen, “MagNet: a Two-Pronged Defense against Adversarial Examples.” [[link]](https://arxiv.org/abs/1705.09064)
  - Use autoencoder to "detect" and "reform" adversarial examples. 
  - Detect: measure probability divergence (Jensen-Shannon divergence + temperature) on classifier's after-softmax output
  - Reform: use output of autoencoder to reconstruct adversarial examples (hoping that it will be reconstructed to the valid distribution)
    - Problem: autoencoder is not perfect as it incurs some reconstruction loss, which is also in some L-p norm. White-box attack can easily obtain the gradient.
- P. Samangouei, M. Kabkab, R. Chellappa, "Defense-GAN: Protecting Classifiers Against Adversarial Attacks Using Generative Models," 2018. [[link]](https://arxiv.org/abs/1805.06605)
  - Use WGAN trained on MNIST (F-MNIST, CelebA) in addition to any black-box neural network, which is trained on the same training set. Works against both white-box and black-box attacks.
  - Use gradient descent to search a latent variable $z$ that produces a sample $G(z)$ closest in L2 distance to a given input $x$, i.e. $\min ||G(z) - x||_2^2$
  - Problems: require GAN that can model the data distribution (almost) perfectly, GD steps add lots of overhead, still vulnerable to on-distribution adversarial examples
- Ilyas et al., "The Robust Manifold Defense: Adversarial Training using Generative Models," 2018.
- L. Schott et al., "Towards the First Adversarially Robust Neural Network Model on MNIST," 2018. [[link]](https://arxiv.org/abs/1805.09190)
  - ABS (Analysis by Synthesis): train VAE for each class of MNIST, use GD to find a point in the latent space that minimizes lower bound of log-likelihood (for VAE's of all classes), finetune with learnable class-dependent scalar, pass to softmax for classification
  -  Greate evaluation: evaluated on different attacks, norms, and with multiple defenses. For each sample, use many attacks and report one with smallest distortion
  -  ABS seems to be the most robust model for all norms ($L_0, L_2, L_\infty$), but successful attacks on ABS have a high variance of distortion
### Beating Defenses
  - N. Carlini and D. Wagner, “Adversarial Examples Are Not Easily Detected: Bypassing Ten Detection Methods.”
  - W. He, J. Wei, X. Chen, and N. Carlini, “Adversarial Example Defenses: Ensembles of Weak Defenses are not Strong.”
  - Athalye et al., "Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples," 2018.
  
---

## Theoretical Analysis
- O. Bastani, Y. Ioannou, L. Lampropoulos, D. Vytiniotis, A. V Nori, and A. Criminisi, “Measuring Neural Net Robustness with Constraints.”
- A. Fawzi, O. Fawzi, and P. Frossard, “Analysis of classifiers’ robustness to adversarial perturbations.”
- A. Matyasko and L. Chau, “Margin Maximization for Robust Classification using Deep Learning,” pp. 300–307, 2017.
- J. Sokolic, R. Giryes, G. Sapiro, and M. R. D. Rodrigues, “Robust Large Margin Deep Neural Networks,” May 2016.
- E. D. Cubuk, B. Zoph, S. S. Schoenholz, Q. V Le, and G. Brain, “INTRIGUING PROPERTIES OF ADVERSARIAL EXAMPLES.”
  - Universal properties of small perturbation
  - Improving robustness with larger logits difference
- Gilmer et al., "Adversarial Spheres," 2018.
  - On-distribution adversarial examples
  - Any none-zero test error leads to adversarial examples 
  
---

## Applications
- Physically robust adversarial examples
  - A. Kurakin, I. J. Goodfellow, and S. Bengio, “ADVERSARIAL EXAMPLES IN THE PHYSICAL WORLD.”
  - A. Athalye, L. Engstrom, A. Ilyas, and K. Kwok, “SYNTHESIZING ROBUST ADVERSARIAL EXAMPLES.”
  - J. Lu, H. Sibai, E. Fabry, and D. Forsyth, “NO Need to Worry about Adversarial Examples in Object Detection in Autonomous Vehicles,” 2017.
  - J. Lu, H. Sibai, E. Fabry, and D. Forsyth, “Standard detectors aren’t (currently) fooled by physical adversarial stop signs.”
  - C. Sitawarin, A. Bhagoji, A. Mosenia, M. Chiang, P. Mittal, "DARTS: Deceiving Autonomous Cars with Toxic Signs," 2018.
- C. Xie, J. Wang, Z. Zhang, Y. Zhou, L. Xie, and A. Yuille, “Adversarial Examples for Semantic Segmentation and Object Detection.”
- S. Huang, N. Papernot, I. Goodfellow, Y. Duan, and P. Abbeel, “Adversarial Attacks on Neural Network Policies.”
- J. Lu, H. Sibai, and E. Fabry, “Adversarial Examples that Fool Detectors.”
### Text
- J. Gao, J. Lanchantin, M. Soffa, Y. Qi, "Black-box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers," 2018.
- Kuleshov et al., "Adversarial Examples for Natural Language Classification Problems," 2018.
- Ebrahimi et al., "HotFlip: White-Box Adversarial Examples for Text Classification," 2018.
- Samanta et al., "Towards Crafting Text Adversarial Samples," 2018.
### Audio
- Carlini et al., "Audio Adversarial Examples: Targeted Attacks on Speech-to-Text," 2018.

---

## Etc.
- Neural network verification
  - X. Huang, M. Kwiatkowska, S. Wang, and M. Wu, “Safety Verification of Deep Neural Networks.”
  - G. Katz, C. Barrett, D. Dill, K. Julian, and M. Kochenderfer, “Reluplex: An Efficient SMT Solver for Verifying Deep Neural Networks.”
  - N. Carlini, G. Katz, C. Barrett, and D. L. Dill, “Ground-Truth Adversarial Examples.”
- Elsayed et al., "Adversarial Examples that Fool both Human and Computer Vision," 2018.

---

## Other Cool Security/Adversarial ML papers
- N. Carlini et al., “Hidden Voice Commands,” USENIX, 2016. [[link]](https://www.usenix.org/conference/usenixsecurity16/technical-sessions/presentation/carlini)
- M. Sharif, S. Bhagavatula, L. Bauer, and M. K. Reiter, "Accessorize to a Crime: Real and Stealthy Attacks on State-of-the-Art Face Recognition," In Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security (CCS '16). ACM, New York, NY, USA, 1528-1540. DOI: [[link]](https://doi.org/10.1145/2976749.2978392)

---

## Useful Links
- https://github.com/tensorflow/cleverhans
- https://evademl.org/
- https://github.com/mzweilin/EvadeML-Zoo
- https://github.com/MadryLab/mnist_challenge

---

## To-Read
- A. Bagnall, R. Bunescu, and G. Stewart, “Training Ensembles to Detect Adversarial Examples,” 2017.
- K. Grosse, D. Pfaff, M. T. Smith, and M. Backes, “How Wrong Am I? — Studying Adversarial Examples and their Impact on Uncertainty in Gaussian Process Machine Learning Models.”
- Z. Sun, M. Ozay, and T. Okatani, “HYPERNETWORKS WITH STATISTICAL FILTERING FOR DEFENDING ADVERSARIAL EXAMPLES.”
- C. Xie, Z. Zhang, A. L. Yuille, J. Wang, and Z. Ren, “MITIGATING ADVERSARIAL EFFECTS THROUGH RANDOMIZATION.”
- K. Grosse, P. Manoharan, N. Papernot, M. Backes, and P. Mcdaniel, “On the (Statistical) Detection of Adversarial Examples.”
- J. Z. Kolter and E. Wong, “Provable defenses against adversarial examples via the convex outer adversarial polytope,” 2017.
- R. Huang, B. Xu, D. Schuurmans, and C. Szepesvári, “LEARNING WITH A STRONG ADVERSARY.”
- P.-Y. Chen, H. Zhang, Y. Sharma, J. Yi, and C.-J. Hsieh, “ZOO: Zeroth Order Optimization based Black-box Attacks to Deep Neural Networks without Training Substitute Models.”
- J. Hayes and G. Danezis, “Machine Learning as an Adversarial Service: Learning Black-Box Adversarial Examples.”
- X. Cao and N. Z. Gong, “Mitigating Evasion Attacks to Deep Neural Networks via Region-based Classification.”
- N. Narodytska and S. Kasiviswanathan, “Simple Black-Box Adversarial Attacks on Deep Neural Networks.”
- H. Hosseini, Y. Chen, S. Kannan, B. Zhang, and R. Poovendran, “Blocking Transferability of Adversarial Examples in Black-Box Learning Systems.”
- T. Pang, C. Du, Y. Dong, and J. Zhu, “Towards Robust Detection of Adversarial Examples.” [[link]](https://arxiv.org/pdf/1706.00633.pdf)
- X. Yuan, P. He, Q. Zhu, R. R. Bhat, and X. Li, “Adversarial Examples: Attacks and Defenses for Deep Learning.” [[link]](https://arxiv.org/pdf/1712.07107.pdf)
- A. Ilyas, L. Engstrom, A. Athalye, and J. Lin, “Query-efficient Black-box Adversarial Examples.” [[link]](https://arxiv.org/pdf/1712.07113.pdf)
- A. Rawat, M. Wistuba, and M.-I. Nicolae, “Harnessing Model Uncertainty for Detecting Adversarial Examples.”[[link]](http://bayesiandeeplearning.org/2017/papers/37.pdf)
- J. Ebrahimi, A. Rao, D. Lowd, and D. Dou, “HotFlip: White-Box Adversarial Examples for NLP.”
- F. Suya, Y. Tian, D. Evans, and P. Papotti, “Query-limited Black-box Attacks to Classifiers.”
