# Adversarial Examples Reading List
A compilation of papers in adversarial examples that I have read or plan to read. The order of the papers is arbitrary. Any paper suggestion is very welcomed.
I recommend using **TeX All the Things** Chrome Extension for viewing math equations on this page.

## Table of Contents
- [Adversarial Examples Reading List](#adversarial-examples-reading-list)
  - [Table of Contents](#table-of-contents)
  - [Background](#background)
  - [Attacks](#attacks)
    - [Attacks with GAN](#attacks-with-gan)
  - [Defenses](#defenses)
    - [Defenses with GAN, VAE](#defenses-with-gan-vae)
    - [Beating Defenses](#beating-defenses)
  - [Theoretical & Empirical Analysis](#theoretical--empirical-analysis)
  - [Applications](#applications)
    - [Text](#text)
    - [Audio](#audio)
  - [Etc.](#etc)
  - [Other Cool Security/Adversarial ML papers](#other-cool-securityadversarial-ml-papers)
  - [Useful Links](#useful-links)
  - [To-Read](#to-read)

---  

## Background

- :+1: &nbsp; C. Szegedy, J. Bruna, D. Erhan, and I. Goodfellow, **Intriguing Properties of Neural Networks**, ICLR 2014. [[link]](https://arxiv.org/abs/1312.6199)
- :+1: &nbsp; I. J. Goodfellow, J. Shlens, and C. Szegedy, **Explaining and Harnessing Adversarial Examples**, ICLR 2015. [[link]](https://arxiv.org/abs/1412.6572) 
- :+1: &nbsp; A. Nguyen, J. Yosinski, and J. Clune, **Deep Neural Networks are Easily Fooled**, CVPR, 2015 IEEE Conf., pp. 427–436, 2015.
- :+1: &nbsp; N. Papernot, P. Mcdaniel, I. Goodfellow, S. Jha, Z. B. Celik, and A. Swami, **Practical Black-Box Attacks against Deep Learning Systems using Adversarial Examples.**
- :+1: &nbsp; N. Papernot, P. McDaniel, S. Jha, M. Fredrikson, Z. B. Celik, and A. Swami, **The Limitations of Deep Learning in Adversarial Settings**, IEEE, Nov. 2015.
- :+1: &nbsp; N. Papernot, P. Mcdaniel, A. Sinha, and M. Wellman, **SoK : Towards the Science of Security and Privacy in Machine Learning.**
- :+1: &nbsp; J. Gilmer et al., **Motivating the Rules of the Game for Adversarial Example Research**, 2018. [[link]](https://arxiv.org/abs/1807.06732)
  - Analyze and propose different threat models for adversarial examples in the context of real-world security-critical applications
  
---

## Attacks

- :+1: &nbsp; Sabour et al., **Adversarial Manipulation of Deep Rrepresentations**, ICLR 2016.
  - Create adversarial examples by matching deep representation of an original sample to that of a guide sample by reducing $\ell_2$ distance of deep representations under a box constraint in the pixel space.
  - Through some analyses, the adversarial examples are found to be __more similar__ to the guide sample than the original despite little change in pixel space. A number of experiments shows that the nature of this deep-representation adversarial examples is very different from that of the normal ones. The experiment with random network weights suggest that this phenomenon might be caused by network architecture rather than the learning algorithm itself.
- :+1: &nbsp; N. Carlini and D. Wagner, **Towards Evaluating the Robustness of Neural Networks**, SP 2017.
- P.-Y. Chen, Y. Sharma, H. Zhang, J. Yi, and C.-J. Hsieh, **EAD: Elastic-Net Attacks to Deep Neural Networks via Adversarial Examples.**
- O. Poursaeed, I. Katsman, B. Gao, and S. Belongie, **Generative Adversarial Perturbations.**
- S. Baluja and I. Fischer, **Adversarial Transformation Networks: Learning to Generate Adversarial Examples.**
  - Train neural net to generate adversarial examples
- F. Tramèr, A. Kurakin, N. Papernot, D. Boneh, and P. Mcdaniel, **Ensemble Adversarial Training: Attacks and Defenses.**
- :+1: &nbsp; Y. Liu, X. Chen, C. Liu, and D. Song, **Delving into Transferable Adversarial Examples and Black-box Attacks**, no. 2, pp. 1–24, 2016.
- :+1: &nbsp; S.-M. Moosavi-Dezfooli, A. Fawzi, O. Fawzi, and P. Frossard, **Universal adversarial perturbations**, 2016.
- :+1: &nbsp; S.-M. Moosavi-Dezfooli, A. Fawzi, and P. Frossard, **DeepFool: a simple and accurate method to fool deep neural networks**, CVPR, pp. 2574–2582, 2016.
- :+1: &nbsp; F. Tramèr, N. Papernot, I. Goodfellow, D. Boneh, and P. Mcdaniel, **The Space of Transferable Adversarial Examples.**
- :+1: &nbsp; M. Cisse, Y. Adi, N. Neverova, and J. Keshet, **Houdini: Fooling Deep Structured Prediction Models.**
  - Generating adversarial examples using surrogate loss in place of real non-differentiable task loss
- :+1: &nbsp; W. Brendel, J. Rauber, and M. Bethge, **Decision-Based Adversarial Attacks: Reliable Attacks Against Black-Box Machine Learning Models**, 2017. [[link]](https://arxiv.org/abs/1712.04248)
  - Attack that requires only the classifier's output (# of queries $\approx10^5$). Start with an image of target class and move towards a desired benign image.
- :+1: &nbsp; Xiao et al., **Spatially Transformed Adversarial Examples**, ICLR 2018. [[link]](https://arxiv.org/abs/1801.02612) [[Github]](https://github.com/rakutentech/stAdv)
  - White-box attack using GD on a different objective function calculated from displacement of pixels (called *flow*), use differentiable bilinear interpolation for continuous (and differentiable) objective function.
- :+1: &nbsp; **The Limitation of Adversarial Training and the Blind-Spot Attack**, ICLR 2019
  - Measure (1) distance from one test sample to all training samples with mean distance of the $k$-nearest neighbors in an arbitrary deep representation, (2) KL divergence of training and test sets (use same deep representation as (1), project with t-SNE, then KDE)
  - Blind-spot attack: find test sample __far away__ from the training samples by pixel-wise affine transform and clipping to [0, 1], then use CW $\ell_\infty$ attack
  - Very successful against adversarial training: the attack is far from the learned manifold, even for adversarial training. Interestingly, the affine transformation part does not affect clean accuracy at all. So the network generalizes to such transformation, but the transformation does put the starting sample in a __more vulnerable__ region.

### Attacks with GAN

- Z. Zhao, D. Dua, and S. Singh, **Generating Natural Adversarial Examples.**
- J. Hayes, G. Danezis, **Learning Universal Adversarial Perturbations with Generative Models.**
- Xiao et al., **GENERATING ADVERSARIAL EXAMPLES WITH ADVERSARIAL NETWORKS**, 2018.
- Poursaeed et al., **Generative Adversarial Perturbations**, 2017.
  
---

## Defenses

- :+1: &nbsp; N. Papernot, P. McDaniel, X. Wu, S. Jha, and A. Swami, **Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks**, 2015.
- :+1: &nbsp; A. Kurakin, G. Brain, I. J. Goodfellow, and S. Bengio, **Adversarial Machine Learning at Scale.**
  - First introduction of adversarial training with FGSM
- S. Gu, L. Rigazio, **Towards Deep Neural Network Architectures Robust to Adversarial Examples**, 2015.
- :+1: &nbsp; A. Mądry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, **Towards Deep Learning Models Resistant to Adversarial Attacks.**
  - Adversarial training with PGD provides strong defense (MNIST, CIFAR) even in white-box setting
- S. Zheng, T. Leung, and I. Goodfellow, **Improving the Robustness of Deep Neural Networks via Stability Training.**
- Hendrik Metzen et al., **On Detecting Adversarial Perturbations**, ICLR 2017.
  - Use a classifier with a branch off one of the deep representations. The branch detects adversarial examples, and the entire network is trained end-to-end with adversarial training using FGSM (clean : adv = 1 : 1).
  - The method works relatively well even against adaptive adversary (FGSM, BIM, DeepFool). There seems to be an optimal magnitude of perturbation. Too small or too large perturbation seem to be easily detected. Detectors trained for large perturbation (FGSM) also seem to not generalize well to adversarial examples with small perturbation. This potentially suggests that this previous adversarial training method is not robust as it fails to generalize. Bypassed by Carlini and Wagner 2017.
- Feinman et al., **Detecting Adversarial Samples from Artifacts**, 2017.
  - Use two metrics, kernel density estimation and uncertainty measure with dropout, to detect adversarial examples relying on a claim that adversarial examples lie outside of a learned manifold resulting in low density region and high uncertainty. Note that both measurements are calculated on a deep representation due to a claim that it is more well-behaved.
  - Gal & Ghahramani (2015): dropout is an approximate deep Gaussian process. So randomly picked distribution on dropout can potentially measure uncertainty of given samples.
  - Combining both metrics, a simple logistic regression can seemingly reliably detect adversarial examples (FGSM, BIM, CW). However, it is bypassed by Carlini and Wagner 2017.
- Xie et al., **Mitigating Adversarial Effects through Randomization**, 2017.
  - Defense against adversarial examples with randomization layer which resizes and pads an image, use ensemble with adversarial training like Tramer et al. 2017.
  - Evaluated on ImageNet against an attack that uses an ensemble of a fixed set of 21 patterns (resize + pad). Seem to improve robustness significantly against gradient-based attack, especially DeepFool and CW. 
  - CW performs slightly worse than FGSM in this case potentially because the randomization further prevents transferability which is already poor for CW. Curious to see an attack directly on the ensemble of models, i.e. the patterns are re-randomized at every iteration.
- :+1: &nbsp; H. Kannan, A. Kurakin, I. Goodfellow, **Adversarial Logit Pairing**, 2018.
- A. Galloway, T. Tanay, G. Taylor, **Adversarial Training Versus Weight Decay**, 2018.
- A. Mosca, and G. Magoulas, **Hardening against adversarial examples with the smooth gradient method**, 2018.
- :+1: &nbsp; A. Raghunathan, J. Steinhardt, and P. Liang, **Certified Defenses against Adversarial Examples**, 2018.
- W. Xu, D. Evans, and Q. Yanjun, **Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks**, NDSS 2018. [[link]](https://arxiv.org/abs/1704.01155)
  - Experiment with three "feature squeezing": reduce bit depth, local smoothing, non-local smoothing.
  - Evaluated on MNIST, CIFAR-10, ImageNet. Some performance drop on CIFAR-10 and ImageNet.
  - Each method works well with different types of norms (i.e. bit depth reduction is very good against $\ell_2$ or $\ell_\infty$, smoothing is good against $\ell_0$, etc.).
  - Can be used as a detector by comparing ($\ell_1$ distance) logits of the input before and after squeezing.
  - Obvious adaptive adversary does not succeed.
- N. Worzyk, and O. Kramer, **Properties of adv−1 – Adversarials of Adversarials**, ESANN 2018. [[link]](https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2018-164.pdf)
  - Apply various adversarial example generation methods on top of adversarial examples, measure $L_2$ difference for detecting adversarial examples
  - A good portion of "doubly adversarial" examples revert to their original correct class, depending on which attack is used first and which is used after
- S. Srisakaokul, Z. Zhong, Y. Zhang, W. Yang, and T. Xie, **MULDEF: Multi-model-based Defense Against Adversarial Examples for Neural Networks**, 2018. [[link]](https://arxiv.org/abs/1809.00065)
  - They propose a simple scheme of adversarial training with multiple models: to summarize, the first model is trained on clean samples, and each of the subsequent models is trained on a union of clean samples and FGSM adversarial examples generated from all of the models before it.
  - The rebustness relies on the random model selection, and each model is not robust to its own adversarial examples but significantly more robust to adversarial examples generated from the other models.

### Defenses with GAN, VAE

- Y. Song at al., **PixelDefend: Leveraging Generative Models to Understand and Defend Against Adversarial Examples**, 2017. [[link]](https://arxiv.org/abs/1710.10766)
- S. Shen et al., **APE-GAN: Adversarial Perturbation Elimination with GAN.**
- :+1: &nbsp; D. Meng and H. Chen, **MagNet: a Two-Pronged Defense against Adversarial Examples.** [[link]](https://arxiv.org/abs/1705.09064)
  - Use autoencoder to "detect" and "reform" adversarial examples.
  - Detect: measure probability divergence (Jensen-Shannon divergence + temperature) on classifier's after-softmax output
  - Reform: use output of autoencoder to reconstruct adversarial examples (hoping that it will be reconstructed to the valid distribution)
    - Problem: autoencoder is not perfect as it incurs some reconstruction loss, which is also in some L-p norm. White-box attack can easily obtain the gradient.
- :+1: &nbsp; P. Samangouei, M. Kabkab, and R. Chellappa, **Defense-GAN: Protecting Classifiers Against Adversarial Attacks Using Generative Models**, 2018. [[link]](https://arxiv.org/abs/1805.06605)
  - Use WGAN trained on MNIST (F-MNIST, CelebA) in addition to any black-box neural network, which is trained on the same training set. Works against both white-box and black-box attacks.
  - Use gradient descent to search a latent variable $z$ that produces a sample $G(z)$ closest in L2 distance to a given input $x$, i.e. $\min ||G(z) - x||_2^2$
  - Problems: require GAN that can model the data distribution (almost) perfectly, GD steps add lots of overhead, still vulnerable to on-distribution adversarial examples
- :+1: &nbsp; Ilyas et al., **The Robust Manifold Defense: Adversarial Training using Generative Models**, 2018.
- :+1: &nbsp; L. Schott et al., **Towards the First Adversarially Robust Neural Network Model on MNIST**, 2018. [[link]](https://arxiv.org/abs/1805.09190)
  - ABS (Analysis by Synthesis): train VAE for each class of MNIST, use GD to find a point in the latent space that minimizes lower bound of log-likelihood (for VAE's of all classes), finetune with learnable class-dependent scalar, pass to softmax for classification
  - Greate evaluation: evaluated on different attacks, norms, and with multiple defenses. For each sample, use many attacks and report one with smallest distortion
  - ABS seems to be the most robust model for all norms ($\ell_0, \ell_2, \ell_\infty$), but successful attacks on ABS have a high variance of distortion
- Jasjeet Dhaliwal, Saurabh Shintre, **Gradient Similarity: An Explainable Approach to Detect Adversarial Attacks against Deep Learning**, 2018.
  - Gradient similarity defined between a pair of train and test samples, taken from influence function (Koh and Liang 2017), but the Hessian is omitted as they show that the Hessian only scales with constant. $GS(x^*,x') = \nabla_\theta L(\theta^*, x', y')^T \cdot \nabla_\theta L(\theta^*, x^*, y^*)$
  - GS is separated into two features: $\ell_2$-norm of gradient and cosine of the angle between the two gradient terms. Logistic regression is trained on the features, half clean and half adversarial (FGSM, BIM, CW, DeepFool, JSMA). Works well against adversaries that have no knowledge of the defense, but fails in white-box setting.

### Beating Defenses

- N. Carlini and D. Wagner, **Adversarial Examples Are Not Easily Detected: Bypassing Ten Detection Methods.**
- W. He, J. Wei, X. Chen, and N. Carlini, **Adversarial Example Defenses: Ensembles of Weak Defenses are not Strong.**
- Athalye et al., **Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples**, 2018.
  
---

## Theoretical & Empirical Analysis

- O. Bastani, Y. Ioannou, L. Lampropoulos, D. Vytiniotis, A. V Nori, and A. Criminisi, **Measuring Neural Net Robustness with Constraints.**
- A. Fawzi, O. Fawzi, and P. Frossard, **Analysis of classifiers’ robustness to adversarial perturbations.**
- A. Matyasko and L. Chau, **Margin Maximization for Robust Classification using Deep Learning**, pp. 300–307, 2017.
- J. Sokolic, R. Giryes, G. Sapiro, and M. R. D. Rodrigues, **Robust Large Margin Deep Neural Networks**, May 2016.
- E. D. Cubuk, B. Zoph, S. S. Schoenholz, Q. V Le, and G. Brain, **Intriguing Properties of Adversarial Examples.**
  - Universal properties of small perturbation
  - Improving robustness with larger logits difference
- :+1: &nbsp; Gilmer et al., **Adversarial Spheres**, 2018.
  - On-distribution adversarial examples
  - Any none-zero test error leads to adversarial examples 
- D. Su, H. Zhang, H. Chen, J. Yi, P.Y. Chen, and Y. Gao, **Is Robustness the Cost of Accuracy? -- A Comprehensive Study on the Robustness of 18 Deep Image Classification Models**, ECCV 2018. [[link]](https://arxiv.org/abs/1808.01688)
  - There is a trade-off between robustness and test accuracy, i.e. $distortion \sim \log(classification~error)$
  - Network architecture affects robustness more than model size
  - Adversarial examples generated from VGG transfer well to the other models while the rest often only transfer within the same family of models. Transferability is often asymmetric.
- Chen and Vorobeychik, **Regularized Ensembles and Transferability in Adversarial Learning**, AAAI 2018.
  - Transferability seems to be blocked by 1) different last layer/loss (normal, SVM), 2) regularization norm ($\ell_1, \ell_2$), 3) regularization constant (5, 4.999999, 4.9999999). Choice of regularization constant is particularly interesting. Could tiny change in the loss function change the landscape of the optima?
  - Attack with ensemble (average last layer): more sub-models increases attack success rate, redundant sub-models in the ensemble (sub-models not in the target) hurts success rate slightly, but after two extra models, adding more does not reduce success rate. This is likely due to the two type of regularizations used or potentially the choice of target network.
- :+1: &nbsp; Jetley et al., **With Friends Like These, Who Needs Adversaries?**, NIPS 2018.
  - Experiment with some __universal class-dependent__ directions in the input space, extending the works of Fawzi et al.: (1) use DeepFool to find boundary close to samples, (2) find Hessian of difference of loss of the correct class and all other classes, (3) find eigenvalues, eigenvectors of mean of Hessian.
  - Moving along or in reverse direction of most positive or most negative curvature (eigenvalue) significantly affects class scores whereas moving along directions with zero curvature does not. These directions are mostly shared between samples, and the ones with high curvature contribute to classification performance but is mutually exploited by adversarial examples (shown by linearly projecting data to subspaces with different numbers of dimension).
  - Hypothesis: attempt to improve robustness by removing directions exploited by adversaries will significantly affect classification performance.
- Schmidt et al., **Adversarially Robust Generalization Requires More Data**, NIPS 2018.
  - Study two simple data distributions (Gaussian and Bernoulli) and size of training set required for __any__ learning algorithm to have robust generalization. The gap between robust and normal generalization is a factor of $O(\sqrt{d})$ for Gaussian. For Bernoulli, with thresholding, the gap is $O(1)$.
  - Experiment on MNIST shows that more training data is more robust, normal < threshold < adversarial training < adversarial training + threshold.
- Wang et al., **One Bit Matters: Understanding Adversarial Examples as the Abuse of Redundancy**, 2018.
  - Consider a simple model: input is $x_1, x_2, x_3$, network outputs $x_1 == x_2$. $x_3$ is thus "redundant" and all neurons connected to it should have weight of zero for the network to make no error, but this constraint on the weights grow exponentially on redundant input.
  - "General model" has two components: erasing noise/unrelated features, and matching with known patterns. Adversarial attacks simply create patterns or enough redundancy that the model cannot completely erase.
  - Hypothesis & experiment: (1) Adversarial examples are more "complex": (a) require larger networks to memorize/classify, (b) higher entropy (i.e. for MNIST, maximum likelihood, minimax, compression estimator). (2) More robust models learn features with less entropy.
- D. Stutz, M. Hein, and B. Schiele, **Disentangling Adversarial Robustness and Generalization**, 2018.
  - Define on- and off-manifold adversarial examples. Data manifold is assumed to be learned by VAE-GAN, or is approximated by subspace formed by some nearest neighbors.
    - What's a good way to approximate manifold locally?
  - Claim that regular (gradient-based) adversarial examples are off manifold by measuring distance between a sample and its projection on the "true manifold." Also claim that regular perturbation is almost orthogonal to the manifold, and  by projecting it back to the manifold almost returns it to the original sample.
  - Adversarial training with on-manifold adversarial examples (i.e. searching in latent space of VAE-GAN) improves "generalization": lower test error, lower on-manifold adv. success rate. Does not lower off-manifold adv. success rate.
  - Regular adversarial training (Madry et al.) does not improve test error and is not robust to on-manifold adversarial examples.
- :+1: &nbsp; Ford et al., **Adversarial Examples Are a Natural Consequence of Test Error in Noise**, 2018.
  - Try to show that adversarial examples are not a surprising phenomenon and can be expected from any well-behaved classifiers. The $\epsilon$-boundary measure, probability that $x$ is $\epsilon$ away from the error set, is large even if  the error set itself is small.
  - They also show, on CIFAR-10 and ImageNet, that training with different types of noises, especially Gaussian with large std, can moderately improve model's robustness against adversarial examples (but less than adversarial training). However, an adversarially trained model is also more susceptible to Gaussian noise but still better than the undefended model.
  - Defenses that exhibit gradient masking (does not create a larger distance to boundary) do not improve robustness to other types of noise. Encourage improvement in robustness to general noise, and say that perfect accuracy under noises is nontrivial.
- :+1: &nbsp; Ding et al., **On the Sensitivity of Adversarial Robustness to Input Data Distribution**, ICLR 2019.
  - Adversarial robustness depends heavily on some characteristic of dataset. Data can be transformed in such a way that its clean accuracy is unchanged, but robust accuracy can vary significantly. Hypothesis: loosely speaking, feature squeezing makes dataset "easier" to learn so adversarial training works better, hence the model becomes more robust.
  - Saturated CIFAR-10 (similar to bit depth squeezing) seems be very robust when combined with adversarial training. Edge detection on FMNIST or adjusting gamma on CIFAR-10 seems to also affect robustness after adversarial training.
  - Analysis on the phenomenon: (1) "perturbable volume": binarization pushes data towards the edge of the allowed domain? Removing the [0, 1] constraint in the attack does not seem to have any effect, though in this case, the volumes inside and outside the training data manifold should be naturally different. (2) Inter-class distance: seems correlated, but there are lots of easy counterexamples. "Distance" here is $\ell_2$-based and seems arbitrary (top 10% nearest neighbors). (3) Model capacity and training set size have significant effects. Larger models are usually more robust but requires more data. More data reduces the generalization gap (between test and train robust accuracies) and the gap between clean and robust accuracies,


---

## Applications

- Physically robust adversarial examples
  - A. Kurakin, I. J. Goodfellow, and S. Bengio, **ADVERSARIAL EXAMPLES IN THE PHYSICAL WORLD.**
  - :+1: &nbsp; A. Athalye, L. Engstrom, A. Ilyas, and K. Kwok, **Synthesizing Robust Adversarial Examples**, 2018. [[link]](https://arxiv.org/abs/1707.07397)
  - J. Lu, H. Sibai, E. Fabry, and D. Forsyth, **NO Need to Worry about Adversarial Examples in Object Detection in Autonomous Vehicles**, 2017.
  - J. Lu, H. Sibai, E. Fabry, and D. Forsyth, **Standard detectors aren’t (currently) fooled by physical adversarial stop signs.**
  - C. Sitawarin, A. Bhagoji, A. Mosenia, M. Chiang, P. Mittal, **DARTS: Deceiving Autonomous Cars with Toxic Signs**, 2018.
- C. Xie, J. Wang, Z. Zhang, Y. Zhou, L. Xie, and A. Yuille, **Adversarial Examples for Semantic Segmentation and Object Detection.**
- S. Huang, N. Papernot, I. Goodfellow, Y. Duan, and P. Abbeel, **Adversarial Attacks on Neural Network Policies.**
- J. Lu, H. Sibai, and E. Fabry, **Adversarial Examples that Fool Detectors.**
  
### Text

- J. Gao, J. Lanchantin, M. Soffa, Y. Qi, **Black-box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers**, 2018.
- Kuleshov et al., **Adversarial Examples for Natural Language Classification Problems**, 2018.
- Ebrahimi et al., **HotFlip: White-Box Adversarial Examples for Text Classification**, 2018.
- Samanta et al., **Towards Crafting Text Adversarial Samples**, 2018.
  
### Audio

- Carlini et al., **Audio Adversarial Examples: Targeted Attacks on Speech-to-Text**, 2018.

---

## Etc.

- Neural network verification
  - X. Huang, M. Kwiatkowska, S. Wang, and M. Wu, **Safety Verification of Deep Neural Networks.**
  - G. Katz, C. Barrett, D. Dill, K. Julian, and M. Kochenderfer, **Reluplex: An Efficient SMT Solver for Verifying Deep Neural Networks.**
  - N. Carlini, G. Katz, C. Barrett, and D. L. Dill, **Ground-Truth Adversarial Examples.**
- Elsayed et al., **Adversarial Examples that Fool both Human and Computer Vision**, 2018.

---

## Other Cool Security/Adversarial ML papers

- N. Carlini et al., **Hidden Voice Commands**, USENIX, 2016. [[link]](https://www.usenix.org/conference/usenixsecurity16/technical-sessions/presentation/carlini)
- M. Sharif, S. Bhagavatula, L. Bauer, and M. K. Reiter, **Accessorize to a Crime: Real and Stealthy Attacks on State-of-the-Art Face Recognition**, In Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security (CCS '16). ACM, New York, NY, USA, 1528-1540. DOI: [[link]](https://doi.org/10.1145/2976749.2978392)

---

## Useful Links

- https://github.com/tensorflow/cleverhans
- https://evademl.org/
- https://github.com/mzweilin/EvadeML-Zoo
- https://github.com/MadryLab/mnist_challenge
- http://adversarial-learning.princeton.edu/
- https://secml.github.io/

---

## To-Read

- A. Bagnall, R. Bunescu, and G. Stewart, **Training Ensembles to Detect Adversarial Examples**, 2017.
- K. Grosse, D. Pfaff, M. T. Smith, and M. Backes, **How Wrong Am I? — Studying Adversarial Examples and their Impact on Uncertainty in Gaussian Process Machine Learning Models.**
- Z. Sun, M. Ozay, and T. Okatani, **HYPERNETWORKS WITH STATISTICAL FILTERING FOR DEFENDING ADVERSARIAL EXAMPLES.**
- C. Xie, Z. Zhang, A. L. Yuille, J. Wang, and Z. Ren, **MITIGATING ADVERSARIAL EFFECTS THROUGH RANDOMIZATION.**
- J. Z. Kolter and E. Wong, **Provable defenses against adversarial examples via the convex outer adversarial polytope**, 2017.
- R. Huang, B. Xu, D. Schuurmans, and C. Szepesvári, **LEARNING WITH A STRONG ADVERSARY.**
- P.-Y. Chen, H. Zhang, Y. Sharma, J. Yi, and C.-J. Hsieh, **ZOO: Zeroth Order Optimization based Black-box Attacks to Deep Neural Networks without Training Substitute Models.**
- J. Hayes and G. Danezis, **Machine Learning as an Adversarial Service: Learning Black-Box Adversarial Examples.**
- X. Cao and N. Z. Gong, **Mitigating Evasion Attacks to Deep Neural Networks via Region-based Classification.**
- N. Narodytska and S. Kasiviswanathan, **Simple Black-Box Adversarial Attacks on Deep Neural Networks.**
- H. Hosseini, Y. Chen, S. Kannan, B. Zhang, and R. Poovendran, **Blocking Transferability of Adversarial Examples in Black-Box Learning Systems.**
- T. Pang, C. Du, Y. Dong, and J. Zhu, **Towards Robust Detection of Adversarial Examples.** [[link]](https://arxiv.org/pdf/1706.00633.pdf)
- X. Yuan, P. He, Q. Zhu, R. R. Bhat, and X. Li, **Adversarial Examples: Attacks and Defenses for Deep Learning.** [[link]](https://arxiv.org/pdf/1712.07107.pdf)
- A. Ilyas, L. Engstrom, A. Athalye, and J. Lin, **Query-efficient Black-box Adversarial Examples.** [[link]](https://arxiv.org/pdf/1712.07113.pdf)
- A. Rawat, M. Wistuba, and M.-I. Nicolae, **Harnessing Model Uncertainty for Detecting Adversarial Examples.**[[link]](http://bayesiandeeplearning.org/2017/papers/37.pdf)
- J. Ebrahimi, A. Rao, D. Lowd, and D. Dou, **HotFlip: White-Box Adversarial Examples for NLP.**
- F. Suya, Y. Tian, D. Evans, and P. Papotti, **Query-limited Black-box Attacks to Classifiers.**
