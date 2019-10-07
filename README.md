# Adversarial Examples Reading List
A compilation of papers in adversarial examples that I have read or plan to read. The number of papers in this subfield can be overwhelming. I hope this list is helpful for anyone who is interested in entering the field or looking for a quick survey. The list is by no means exhaustive so any paper suggestion is very welcome. The order of the papers is arbitrary. Couple of notes:
- :+1: simply denotes papers that, solely based on my __personal__ opinion, are influential to the way I think about adversarial examples.
- Some papers will have a short summary I wrote to remind myself what the paper is about + my personal thoughts/reactions. It is not a complete description of the paper.
- I recommend using **TeX All the Things** Chrome Extension for viewing math equations on this page.
- Do expect mistakes, misunderstanding, and shallow interpretation as I am an ignorant PhD student. Please correct me if you found one of those mistakes. I will really appreciate it.
- Enjoy!

## Table of Contents
* [Background](#background)
* [Attacks](#attacks)
  + [Attacks with GAN](#attacks-with-gan)
* [Transferability (Attack and Defense)](#transferability--attack-and-defense-)
* [Defenses](#defenses)
  + [Heuristic Defense](#heuristic-defense)
  + [Detection](#detection)
  + [Certifiable Defense](#certifiable-defense)
  + [Certifiable Defense with "Randomized Smoothing"](#certifiable-defense-with--randomized-smoothing-)
  + [Lipschitz Network](#lipschitz-network)
  + [Defenses with GAN, VAE](#defenses-with-gan--vae)
  + [Ensemble-Based Defense](#ensemble-based-defense)
  + [Beating Defenses](#beating-defenses)
* [Theoretical & Empirical Analysis](#theoretical---empirical-analysis)
  + [Hardness of Defense](#hardness-of-defense)
* [Applications](#applications)
  + [Text](#text)
  + [Audio](#audio)
* [Etc.](#etc)
* [Other Cool Security/Adversarial ML papers](#other-cool-security-adversarial-ml-papers)
* [Useful Links](#useful-links)
* [To-Read](#to-read)

---  

## Background

- :+1: &nbsp; C. Szegedy, J. Bruna, D. Erhan, and I. Goodfellow, **Intriguing Properties of Neural Networks**, ICLR 2014. [[link]](https://arxiv.org/abs/1312.6199)
- :+1: &nbsp; I. J. Goodfellow, J. Shlens, and C. Szegedy, **Explaining and Harnessing Adversarial Examples**, ICLR 2015. [[link]](https://arxiv.org/abs/1412.6572)
- :+1: &nbsp; A. Nguyen, J. Yosinski, and J. Clune, **Deep Neural Networks are Easily Fooled**, CVPR, 2015 IEEE Conf., pp. 427–436, 2015.
- :+1: &nbsp; N. Papernot, P. Mcdaniel, I. Goodfellow, S. Jha, Z. B. Celik, and A. Swami, **Practical Black-Box Attacks against Deep Learning Systems using Adversarial Examples.**
- :+1: &nbsp; N. Papernot, P. McDaniel, S. Jha, M. Fredrikson, Z. B. Celik, and A. Swami, **The Limitations of Deep Learning in Adversarial Settings**, IEEE, Nov. 2015.
- :+1: &nbsp; N. Papernot, P. Mcdaniel, A. Sinha, and M. Wellman, **SoK : Towards the Science of Security and Privacy in Machine Learning.**

---

## Attacks

- :+1: &nbsp; Sabour et al., **Adversarial Manipulation of Deep Rrepresentations**, ICLR 2016.
  - Create adversarial examples by matching deep representation of an original sample to that of a guide sample by reducing $\ell_2$ distance of deep representations under a box constraint in the pixel space.
  - Through some analyses, the adversarial examples are found to be __more similar__ to the guide sample than the original despite little change in pixel space. A number of experiments shows that the nature of this deep-representation adversarial examples is very different from that of the normal ones. The experiment with random network weights suggest that this phenomenon might be caused by network architecture rather than the learning algorithm itself.
- :+1: &nbsp; N. Carlini and D. Wagner, **Towards Evaluating the Robustness of Neural Networks**, IEEE SP 2017.
- P.-Y. Chen, Y. Sharma, H. Zhang, J. Yi, and C.-J. Hsieh, **EAD: Elastic-Net Attacks to Deep Neural Networks via Adversarial Examples.**
- O. Poursaeed, I. Katsman, B. Gao, and S. Belongie, **Generative Adversarial Perturbations.**
- S. Baluja and I. Fischer, **Adversarial Transformation Networks: Learning to Generate Adversarial Examples.**
  - Train neural net to generate adversarial examples
- F. Tramèr, A. Kurakin, N. Papernot, D. Boneh, and P. Mcdaniel, **Ensemble Adversarial Training: Attacks and Defenses.**
- :+1: &nbsp; S.-M. Moosavi-Dezfooli, A. Fawzi, O. Fawzi, and P. Frossard, **Universal adversarial perturbations**, 2016.
- :+1: &nbsp; S.-M. Moosavi-Dezfooli, A. Fawzi, and P. Frossard, **DeepFool: a simple and accurate method to fool deep neural networks**, CVPR, pp. 2574–2582, 2016.
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
- Dong et al., **Boosting Adversarial Attacks with Momentum**, CVPR 2018.
  - Iterative $\ell_\infty$ gradient attack with momentum update. Improve transferability without hurting white-box success rate, but the increase in success rate might be attributed to larger $\ell_2$ norm.


### Attacks with GAN

- Z. Zhao, D. Dua, and S. Singh, **Generating Natural Adversarial Examples.**
- J. Hayes, G. Danezis, **Learning Universal Adversarial Perturbations with Generative Models.**
- Xiao et al., **GENERATING ADVERSARIAL EXAMPLES WITH ADVERSARIAL NETWORKS**, 2018.
- Poursaeed et al., **Generative Adversarial Perturbations**, 2017.

---

## Transferability (Attack and Defense)

- :+1: &nbsp; Y. Liu, X. Chen, C. Liu, and D. Song, **Delving into Transferable Adversarial Examples and Black-box Attacks**, 2016.
- :+1: &nbsp; F. Tramèr, N. Papernot, I. Goodfellow, D. Boneh, and P. Mcdaniel, **The Space of Transferable Adversarial Examples.**
- Xie et al., **Improving Transferability of Adversarial Examples with Input Diversity**, 2018.
  - Random resizing and padding input at every step increases transferability.
- Chen and Vorobeychik, **Regularized Ensembles and Transferability in Adversarial Learning**, AAAI 2018.
  - Transferability seems to be blocked by 1) different last layer/loss (normal, SVM), 2) regularization norm ($\ell_1, \ell_2$), 3) regularization constant (5, 4.999999, 4.9999999). Choice of regularization constant is particularly interesting. Could tiny change in the loss function change the landscape of the optima?
  - Attack with ensemble (average last layer): more sub-models increases attack success rate, redundant sub-models in the ensemble (sub-models not in the target) hurts success rate slightly, but after two extra models, adding more does not reduce success rate. This is likely due to the two type of regularizations used or potentially the choice of target network.
- Shumailov et al., **Sitatapatra: Blocking the Transfer of Adversarial Examples**, 2019.
  - Each convolutional layer is extended with two components: Detector and Guard.
    - Detector: put each activation through element-wise polynomial (coefficients are randomized and kept secret). The model is trained to minimize this polynomial to be under some threshold $t$. Assumption is clean samples won't trigger this threshold, but adversarial examples may.
      - Choice of coefficient affects transferability (more similar more transfer) e.g. $2x^2 + 3x + 5 < t = 6$. But wouldn't this just tries to make activation small? How does it help distinguish OOD? Why not use higher degree?
    - Guard: use some per-channel attention masking with randomized secret pre-determined coefficients. This is said to diversify the gradient and reduce transferability.
  - (Results) weak evaluation section (lack comparison with other methods and baseline). It seems to prevent transfer when $\epsilon$ is small, fail when $\epsilon$ is large but become more detectable.
  - Propose an idea of tracing from which model an adversarial example is generated from. The experiments and evaluations are weak. However, this is an interesting question that no one thought about probably due to lack of real use cases.


---

## Defenses

### Heuristic Defense
- :+1: &nbsp; N. Papernot, P. McDaniel, X. Wu, S. Jha, and A. Swami, **Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks**, 2015.
- :+1: &nbsp; A. Kurakin, G. Brain, I. J. Goodfellow, and S. Bengio, **Adversarial Machine Learning at Scale.**
  - First introduction of adversarial training with FGSM
- S. Gu, L. Rigazio, **Towards Deep Neural Network Architectures Robust to Adversarial Examples**, 2015.
- :+1: &nbsp; A. Mądry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, **Towards Deep Learning Models Resistant to Adversarial Attacks.**
  - Adversarial training with PGD provides strong defense (MNIST, CIFAR) even in white-box setting
- Xie et al., **Mitigating Adversarial Effects through Randomization**, 2017.
  - Defense against adversarial examples with randomization layer which resizes and pads an image, use ensemble with adversarial training like Tramer et al. 2017.
  - Evaluated on ImageNet against an attack that uses an ensemble of a fixed set of 21 patterns (resize + pad). Seem to improve robustness significantly against gradient-based attack, especially DeepFool and CW.
  - CW performs slightly worse than FGSM in this case potentially because the randomization further prevents transferability which is already poor for CW. Curious to see an attack directly on the ensemble of models, i.e. the patterns are re-randomized at every iteration.
- :+1: &nbsp; H. Kannan, A. Kurakin, I. Goodfellow, **Adversarial Logit Pairing**, 2018.
- W. Xu, D. Evans, and Q. Yanjun, **Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks**, NDSS 2018. [[link]](https://arxiv.org/abs/1704.01155)
  - Experiment with three "feature squeezing": reduce bit depth, local smoothing, non-local smoothing.
  - Evaluated on MNIST, CIFAR-10, ImageNet. Some performance drop on CIFAR-10 and ImageNet.
  - Each method works well with different types of norms (i.e. bit depth reduction is very good against $\ell_2$ or $\ell_\infty$, smoothing is good against $\ell_0$, etc.).
  - Can be used as a detector by comparing ($\ell_1$ distance) logits of the input before and after squeezing.
  - Obvious adaptive adversary does not succeed.
- S. Zheng, T. Leung, and I. Goodfellow, **Improving the Robustness of Deep Neural Networks via Stability Training.**
- A. Galloway, T. Tanay, G. Taylor, **Adversarial Training Versus Weight Decay**, 2018.
- A. Mosca, and G. Magoulas, **Hardening against adversarial examples with the smooth gradient method**, 2018.
- S. Srisakaokul, Z. Zhong, Y. Zhang, W. Yang, and T. Xie, **MULDEF: Multi-model-based Defense Against Adversarial Examples for Neural Networks**, 2018. [[link]](https://arxiv.org/abs/1809.00065)
  - They propose a simple scheme of adversarial training with multiple models: to summarize, the first model is trained on clean samples, and each of the subsequent models is trained on a union of clean samples and FGSM adversarial examples generated from all of the models before it.
  - The robustness relies on the random model selection, and each model is not robust to its own adversarial examples but significantly more robust to adversarial examples generated from the other models.
- Mustafa et al., **Adversarial Defense by Restricting the Hidden Space of Deep Neural Networks**, 2019.
  - Propose a defense against adversarial examples, classifying samples by finding distance to "class mean" in the representation layer. The class mean is not really a mean but a trainable weight that is trained as an auxiliary loss (pull: mean & samples from same class. push: mean & samples from diff class, mean & mean from diff class) in addition to normal CE loss. Further improve robustness when combined with PGD/FGSM adversarial training.
  - The results are however not convincing. This method should perform much worse than Madry et al. on MNIST at $\epsilon = 0.3$, but the paper omits this result. Results on CIFAR-10 are also not SOTA.
- Khoury and Hadfield-Menell, **Adversarial Training with Voronoi Constraints**, 2019.
  - Assuming the "low-dimensional manifold hypothesis" (data is $k$-dimensional manifold in $d$-dimensional space where $k \ll d$), they show that kNN is more robust to adversarial examples and is more data-efficient to do "adversarial training"
  - "Nearest neighbor classifiers are naturally robust in high __codimensions__ because the Voronoi cells are elongated in the directions normal to the manifold when data are dense"
  - Theorem 2: Provide sampling condition on training data (there must exist a sample within $\delta$ distance from any point in manifold) such that an $\epsilon$-expansion of the data manifold is classified correctly by nearest neighbor or adversarial training.
  - Theorem 3: The size of training data (under sampling condition from Theorem 2) for nearest neighbor is exponentially smaller in $k$ compared to adversarial training. This proof, however, assumes very simplified manifolds.
  - Propose adversarial training with Voronoi cell constraint instead of a norm ball. The inner maximization problem is similar to that of Madry et al's but constrains that the objective variable, $\hat{x}$, stays in the Voronoi cell of the original training sample $x$. To avoid projection step on Voronoi cell, they take gradient step on the loss function and check if the constraint is satisfied after every step. If not, the optimization terminates. To speed up, instead of checking against the entire training set, only $m$ nearest samples from each class are sought and checked per one training sample, and this needs to be computed only once per dataset.
  - The proposed adversarial training is for neural networks and not nearest neighbor, and no theoretical justification is given for why this adversarial training is better. My take is that it tries to make the network have similar behavior to 1-NN. No need to set $\epsilon$ as it leaves that to property of the dataset. Experiments on MNIST show improve in robustness at large $\ell_\infty$-norm with minimal accuracy drop on clean data. On CIFAR-10, it performs similarly to Madry et al.

### Detection
- Metzen et al., **On Detecting Adversarial Perturbations**, ICLR 2017.
  - Use a classifier with a branch off one of the deep representations. The branch detects adversarial examples, and the entire network is trained end-to-end with adversarial training using FGSM (clean : adv = 1 : 1).
  - The method works relatively well even against adaptive adversary (FGSM, BIM, DeepFool). There seems to be an optimal magnitude of perturbation. Too small or too large perturbation seem to be easily detected. Detectors trained for large perturbation (FGSM) also seem to not generalize well to adversarial examples with small perturbation. This potentially suggests that this previous adversarial training method is not robust as it fails to generalize. Bypassed by Carlini and Wagner 2017.
- Feinman et al., **Detecting Adversarial Samples from Artifacts**, 2017.
  - Use two metrics, kernel density estimation and uncertainty measure with dropout, to detect adversarial examples relying on a claim that adversarial examples lie outside of a learned manifold resulting in low density region and high uncertainty. Note that both measurements are calculated on a deep representation due to a claim that it is more well-behaved.
  - Gal & Ghahramani (2015): dropout is an approximate deep Gaussian process. So randomly picked distribution on dropout can potentially measure uncertainty of given samples.
  - Combining both metrics, a simple logistic regression can seemingly reliably detect adversarial examples (FGSM, BIM, CW). However, it is bypassed by Carlini and Wagner 2017.
- Ma et al., **Characterizing Adversarial Subspaces Using Local Intrinsic Dimensionality**, ICLR 2018. [[link]](https://arxiv.org/abs/1801.02613)
- N. Worzyk, and O. Kramer, **Properties of adv−1 – Adversarials of Adversarials**, ESANN 2018. [[link]](https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2018-164.pdf)
  - Apply various adversarial example generation methods on top of adversarial examples, measure $\ell_2$ difference for detecting adversarial examples
  - A good portion of "doubly adversarial" examples revert to their original correct class, depending on which attack is used first and which is used after
- Dathathri et al., **Detecting Adversarial Examples via Neural Fingerprinting**, 2018 (ICLR 2019 submission). [[link]](https://openreview.net/forum?id=SJekyhCctQ)
  - Train normalized logits to match with some fingerprints: $(\Delta x, \Delta y)$: $\min ||(f(x + \Delta x) - f(x)) - \Delta y||_2^2$. $\Delta x$ is randomly sampled.
  - The detection accuracy is close to perfect even against adaptive white-box adversary using various attack methods (FGSM, PGD, CW, SPAS).
  - The method, however, shows no performance decline even with larger perturbation.

### Certifiable Defense
- :+1: &nbsp; A. Raghunathan, J. Steinhardt, and P. Liang, **Certified Defenses against Adversarial Examples**, 2018.
- :+1: &nbsp; Wong and Kolter, **Provable defenses against adversarial examples via the convex outer adversarial polytope**, ICML 2018. [[link]](https://arxiv.org/abs/1711.00851)
- Wong et al., **Scaling provable adversarial defenses**, NeurIPS 2018. [[link]](https://arxiv.org/abs/1805.12514)

### Certifiable Defense with "Randomized Smoothing"
- :+1: &nbsp; Lecuyer et al., **Certified Robustness to Adversarial Examples with Differential Privacy**, IEEE S&P 2018.
  - Prove robustness lower bound with differential privacy scheme ($\epsilon,\delta$)-DP. If $p_i(x) \geq e^{2\epsilon}p_j(x) + (1+e^\epsilon)\delta$ for $i$ is a correct label and $j \neq i$, then $x$ is robust within an $\ell_p$-ball of radius 1. Use MC to estimate $p(x)$.
  - Gaussian (or Laplacian) noise is added after the first layer. $\sigma \sim \Delta_{p,2}L/\epsilon$. For larger ($L$) and tighter ($\epsilon$) bound, std also needs to be large, but larger std means lower accuracy. $\Delta_{p,2}$ is sensitivity of the first layer (i.e. $p=2$, spectral norm of the weight). Empirically, adding noise after the first layer yields higher accuracy but smaller bound.
  - After noise layer remains DP due to post-processing property of DP.
- Li et al., **Certified Adversarial Robustness with Additive Gaussian Noise**, 2019.
  - Prove certifiable lower bound from Renyi divergence. Estimate probability (and confidence interval) with MC similarly to Lecuyer et al.
  - Use stability training (Zheng et al. '16) to make network perform better when added with  Gaussian noise. Stability training helps significantly. The bound is tighter than Lecuyer et al. Empirical robustness is also decent. Empirical attacks can be better than TRADES for $\ell_2$-adv.
- :+1: &nbsp; Pinot et al., **Theoretical evidence for adversarial robustness through randomization: the case of the Exponential family**, 2019. [[link]](https://arxiv.org/abs/1902.01148)
  - Define robustness for output of neural networks (any layer) as a probabilistic mapping. Rough idea: given some norm-bounded perturbation, measure distance (Renyi divergence) on two random distributions mapped from a clean input and its perturbed version. Ball-bounded input -> (Theorem 3) Renyi-robust after first layer -> (Lemma 1 + Theorem 2) Renyi-div in output -> (Theorem 1) Total Variation distance in output -> certifiable robustness. Only consider noise added to activation after the first layer (same as Lecuyer et al.).
  - "Renyi-robustness" depends on 4 parameters $\lambda, \alpha, \epsilon, \gamma$, defined over distribution of all samples. $\gamma$ is set to 0.
  - Point out that robustness definition (bounded change in output) is disconnected from clean accuracy (i.e. you can have very robust classifier with useless accuracy). Regard noise injection during training as distribution shift problem from noise injection during inference.
  - Evaluate with four distributions from Exponential family (Gaussian, Laplacian, Exponential, Weibull). Exponential seems to do best, but none helps against CW attack (perturbation is unbounded). Assuming one noise draw per sample since there's no need to compute any probability (?). The results are not reported with $\epsilon$ or $\alpha$ so it is difficult to see how the accuracy drops. Clean acc. and $\ell_2$-PGD are very close for the given norm of 0.3.
- :+1: &nbsp; Cohen et al., **Certified Adversarial Robustness via Randomized Smoothing**, 2019. [[link]](https://arxiv.org/abs/1902.02918)

### Lipschitz Network
- :+1: &nbsp; Cisse et al., **Parseval Networks: Improving Robustness to Adversarial Examples**, ICML 2017.
- Tsuzuku et al., **Lipschitz-Margin Training: Scalable Certification of Perturbation Invariance for Deep Neural Networks**, NeurIPS 2018.
- Ono et al., **Lightweight Lipschitz Margin Training for Certified Defense against Adversarial Examples**, 2018.
- Anil et al., **Sorting out Lipschitz Function Approximation**, 2018 (ICLR 2019 submission). [[link]](https://openreview.net/forum?id=ryxY73AcK7)
  - Introduce GroupSort (partition and sort) as an activation function for neural network that has its weight constrained by Lipschitz. They show that ReLU (or tanh) norm-constrained (1-Lipschitz) network cannot approximate universal Lipschitz function (e.g. absolute value, which is 1-Lipschitz). They show theoretically and empirically that GroupSort with unit spectral norm matrix can represent absolute value.
  - They show that in training MNIST classifier with $K = 10$ GroupSort network utilizes "Lipschitz-ness" better (i.e. spectral norm of Jacobian at input $x$ is closer to $K$) than ReLU network.
  - The weight normalization (spectral norm constraint) is done by Bjorck et al. for $\ell_2$ (equivalent to enforcing orthonormality) and by Condat for $\ell_\infty$ case. There is little difference in clean accuracy between ReLU and GroupSort network. Normal network with dropout tends to perform slightly better.
  - Lipschitz network can certify adversarial robustness but in this work, the bound is still slightly loose. Robustness of MNIST is still far inferior compared to Madry et al. (~40% to ~95% accuracy at $\epsilon = 0.3$)
- :+1: &nbsp; Huster et al., **Limitations of the Lipschitz Constant as a Defense Against Adversarial Examples**, 2018.
  - Prove a tighter perturbation bound with Lipschitz constant in binary case: assume that any pair of samples from two different classes is at least $c$ apart in $\ell_p$, there exists a $2/c$-Lipschitz function $f$ such that $sign(f(x+\delta))=y$ for $||\delta||_p < c/2$. An example of such Lipschitz function is given.
  - The bound is, however, coupled with distance $c$ of the dataset. This assumption must be required for any K-Lipschitz bound that tries to achieve 100% accuracy. What's an effective way to balance clean accuracy and robustness with Lipschitz?
  - Given a fully-connected network with ReLU activation, they also show that "atomic" Lipschitz constant, $k_A$ (product of weight norm from each layer) is limiting. It suffers accuracy loss and cannot express some functions (e.g. absolute function which has $k=1$ cannot be expressed by a network with $k_A<2$.
  - Theoretical results make sense, but in practice, this might come down to training methods and how the Lipschitz constant is bounded/penalized. It might be less limiting for real datasets with appropriate techniques.
  - They show that "paired-layer" Lipschitz bound is tighter. This method is used by Raghunathan et al. on two-layer network: $k \leq \max_s ||W_2diag(s)W_1||$ where $s$ is a binary vector indicating which ReLU is on/off. While the bound is tighter in this one (e.g. works with absolute function), it is still not perfect.
  - GroupSort (Anil et al.) tackles this problem differently by using a different activation function completely. GroupSort allows this bound to be tight, but it seems to still hit the fundamental limit in the first assumption.
- Qian & Wegman, **L2-Nonexpansive Neural Networks**, ICLR 2019.
  - Proposes network architecture that preserves $\ell_2$-norm ("distance") of input and output, i.e. Lipschitz from input to logits is less than or equal to 1 under $\ell_2$-norm. To constrain the weight's spectral norm, $||W^TW||_2 \leq 1$, use an $\ell_\infty$ upper bound. ReLU is replaced with two-sided ReLU, and max-pooling is replaced with norm-pooling. The loss function is complicated, consisting of three terms.
  - The weight constraint itself is more general than that of Parseval network.
  - Adversarial robustness is evaluated against CW L2 attacks. The $\ell_2$-norm is bounded at 3 for MNIST and 1.5 for CIFAR-10, and the model seems more robust than $\ell_\infty$ Madry model. Not all attacks succeed at this perturbation, and combining with adv. train further improves robustness slightly.
  - Questions: not sure why norm of Jacobian is not bounded to 1 for "multi-L2NNN classifiers" (page 6)? If it is bounded, can it provide certifiable robustness?

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
  - Gradient similarity defined between a pair of train and test samples, taken from influence function (Koh and Liang 2017), but the Hessian is omitted as they show that the Hessian only scales with constant. $GS(\hat{x},x') = \nabla_\theta L(\hat{\theta}, x', y')^T \cdot \nabla_\theta L(\hat{\theta}, \hat{x}, \hat{y})$
  - GS is separated into two features: $\ell_2$-norm of gradient and cosine of the angle between the two gradient terms. Logistic regression is trained on the features, half clean and half adversarial (FGSM, BIM, CW, DeepFool, JSMA). Works well against adversaries that have no knowledge of the defense, but fails in white-box setting.
- Kyatham et al., **Variational Inference with Latent Space Quantization for Adversarial Robustness**, 2019.
  - Train a VAE that uses mean squared error instead of log-likelihood for reconstruction and penalize norm of input-latent Jacobian to control Lipschitz constant. The latent space is also quantized "to prevent gradient-based attack." This VAE is used to reconstruct any input sample before passing it to any (undefended) classifier.
  - The main idea is constrained Lipschitz constant will "bound" change in the latent code given change in the input, and so by quantizing the latent space, the perturbation will be ignored by the decoder. There is some flaws, however. First, quantization helps against $\ell_\infty$-norm perturbation, but in the paper, spectral norm is used. This choice should be consistent but unfortunately is specific to adversary.
  - The white-box attack experiments states that quantization "prevents attacks" but does not explain how the attack is carried out. The defense can be better evaluated by replacing the hard quantization with a soft one for gradient-based attack. Also, perturbation norm for CW or Deepfool is not reported. The defense is surprisingly most vulnerable to FGSM. The black-box attack uses transfer between different classifiers.

### Ensemble-Based Defense
- Strauss et al., **Ensemble Methods as a Defense to Adversarial Perturbations Against Deep Neural Networks**, 2018 (ICLR 2018 submission).
  - Experiment with different methods of ensemble (random init, different models, bagging, Gaussian noise) and evaluate their robustness against adversarial examples. Gaussian noise seems to perform the best, but it may still require more experiments to be conclusive.
- Adam et al., **Stochastic Combinatorial Ensembles for Defending Against Adversarial Examples**, 2018.
  - Autoencoder at each layer
  - Orthogonal gradient training
- Grefenstette et al., **Strength in Numbers: Trading-off Robustness and Computation via Adversarially-Trained Ensembles**, 2018 (ICLR 2019 submission).
  - Ensemble two models and then adversarially train them (not an ensemble of two separate adversarially trained models) performs better than single-model adversarial training (even with same number of parameters) and ensemble of two adversarially trained models.
- Pang et al., **Improving Adversarial Robustness via Promoting Ensemble Diversity**, 2019.
  - Train ensemble (average logits) with two regularization terms: (1) increase entropy of averaged output $H(\sum F_i)$, (2) increase "diversity" of non-maximal prediction (via determinant point process) which maximizes at 1 when all non-maximal predictions of all models in ensemble are mutually orthogonal.
  - In white-box setting, experiments show that it increases robustness but slightly less than PGD adversarial training. When combined, it can improve PGD slightly. It also reduces transferability between models in the same ensemble.
- Kariyappa and Qureshi, **Improving Adversarial Robustness of Ensembles with Diversity Training**, 2019.
  - Rely on the claim that adversarial examples in ensemble exist because they share a large adversarial subspace. They claim that reducing gradient alignment between models in an ensemble makes it more robust.
    - It is not clear how reducing the adversarial subspace would provide adversarial robustness. In adversarial case, robustness depends only on the worst-case (closest) point that is misclassified. The method does not seem to affect the boundary so white-box robustness is questionable. It potentially could make adversarial examples harder to find and maybe reduce transferability.
    - It is also not clear how misaligning gradient direction at each sample will reduce the volume of this space.
  - Train ensemble with regularization term penalizing cosine similarity between gradients in the ensemble (smoothed max with logistic). Use leaky ReLU to prevent vanishing gradient (otherwise, gradients are very sparse).
  - Improve accuracy against transfer attack (ensemble of 5 models, architecture is known, only exact weights are not) considerably for small $\epsilon$, but little improvement for large $\epsilon$. Improvement is smaller than Tramer et al. 2017, but can improve a little when combined.

### Beating Defenses
- N. Carlini and D. Wagner, **Adversarial Examples Are Not Easily Detected: Bypassing Ten Detection Methods.**
- W. He, J. Wei, X. Chen, and N. Carlini, **Adversarial Example Defenses: Ensembles of Weak Defenses are not Strong.**
- Athalye et al., **Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples**, 2018.

---

## Theoretical & Empirical Analysis

- O. Bastani, Y. Ioannou, L. Lampropoulos, D. Vytiniotis, A. V Nori, and A. Criminisi, **Measuring Neural Net Robustness with Constraints.**
- Weng et al., **Evaluating the Robustness of Neural Networks: An Extreme Value Theory Approach**, ICLR 2018. [[link]](https://openreview.net/forum?id=BkUHlMZ0b)
  - Attempts to provide model-agnostic robustness metric as empirical robustness heavily depends on attack algorithms and models. CLEVER approximates lower bound of adversarial perturbation (by approximating local Lipschitz constant around a given sample).
  - Approximating the Lipschitz constant is done by Extreme Value Theory, or more precisely Fisher-Tippett-Gnedenko Theorem. For a given sample, CLEVER samples a large number of points in the norm ball around it, computes gradients, and estimates the Lipschitz constant as Reverse Weibull distribution. The experiment also shows that the chosen Reverse Weibull distribution is a "good fit."
  - While the CLEVER bound is mostly correct (no attack finds perturbation smaller than the bound), it is very loose even when compared with supposedly close-to-optimal attack like CW. CLEVER takes reasonable time to run (< 10s per samples) for a large network.
  - Gradient Masking Causes CLEVER to Overestimate Adversarial Perturbation Size: https://arxiv.org/pdf/1804.07870.pdf. A follow-up work: On Extensions of CLEVER: A Neural Network Robustness Evaluation Algorithm https://arxiv.org/pdf/1810.08640.pdf
- A. Matyasko and L. Chau, **Margin Maximization for Robust Classification using Deep Learning**, pp. 300–307, 2017.
- J. Sokolic, R. Giryes, G. Sapiro, and M. R. D. Rodrigues, **Robust Large Margin Deep Neural Networks**, May 2016.
- E. D. Cubuk, B. Zoph, S. S. Schoenholz, Q. V Le, and G. Brain, **Intriguing Properties of Adversarial Examples.**
  - Universal properties of small perturbation
  - Improving robustness with larger logits difference
- D. Su, H. Zhang, H. Chen, J. Yi, P.Y. Chen, and Y. Gao, **Is Robustness the Cost of Accuracy? -- A Comprehensive Study on the Robustness of 18 Deep Image Classification Models**, ECCV 2018. [[link]](https://arxiv.org/abs/1808.01688)
  - There is a trade-off between robustness and test accuracy, i.e. $distortion \sim \log(classification~error)$
  - Network architecture affects robustness more than model size
  - Adversarial examples generated from VGG transfer well to the other models while the rest often only transfer within the same family of models. Transferability is often asymmetric.
- :+1: &nbsp; Jetley et al., **With Friends Like These, Who Needs Adversaries?**, NeurIPS 2018.
  - Experiment with some __universal class-dependent__ directions in the input space, extending the works of Fawzi et al.: (1) use DeepFool to find boundary close to samples, (2) find Hessian of difference of loss of the correct class and all other classes, (3) find eigenvalues, eigenvectors of mean of Hessian.
  - Moving along or in reverse direction of most positive or most negative curvature (eigenvalue) significantly affects class scores whereas moving along directions with zero curvature does not. These directions are mostly shared between samples, and the ones with high curvature contribute to classification performance but is mutually exploited by adversarial examples (shown by linearly projecting data to subspaces with different numbers of dimension).
  - Hypothesis: attempt to improve robustness by removing directions exploited by adversaries will significantly affect classification performance.
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
  - Analysis on the phenomenon: (1) "perturbable volume": binarization pushes data towards the edge of the allowed domain? Removing the [0, 1] constraint in the attack does not seem to have any effect, though in this case, the volumes inside and outside the training data manifold should be naturally different. (2) Inter-class distance: seems correlated, but there are lots of easy counterexamples. "Distance" here is $\ell_2$-based and seems arbitrary (top 10% nearest neighbors). (3) Model capacity and training set size have significant effects. Larger models are usually more robust but requires more data. More data reduces the generalization gap (between test and train robust accuracies) and the gap between clean and robust accuracies.
- Kim et al., **Bridging Adversarial Robustness and Gradient Interpretability**, ICLR Workshop 2019.
  - Hypothesis 1: Adversarial examples on adversarially trained models lie closer to natural data manifold compared to undefended models. They show this using VAE-GAN and measure distance between input and its reconstruction. They also speculate that gradient of samples closer to clean data manifold is more interpretable.
  - Hypothesis 2: Connection to tilting boundary theory. They show on 2D toy dataset that adversarial training aligns the decision boundary more perpendicular to high-variance direction (invariant to low-variance direction), and so they hypothesize that gradient required to cross the boundary aligns better with "human perception."
  - They test two attribution techniques (just gradient and gradient * input) and measure their interpretability score with two metrics (retain and retrain, keep and retrain). They show that strength of adversarial training correlates with higher interpretability scores.
- :+1: &nbsp; Rajput et al., **Does Data Augmentation lead to Positive Margin?**, 2019.
  - Note that all proofs are based on __worst-case margin__ (w.r.t. samples and possible classifier that (non)linearly separates data). This work proves lower bound on the number of augmented data points are needed to increase the margin in various settings (linear/non-linear classifier, norm-bounded/random noise).
  - Theorem 10 (non-linear classifier, non-random augment): For $r \leq \epsilon$, there exists an $r$-ball augmented data set of size $d+1$ times the original, that increases margin for any classifier chosen from a set of $\epsilon$__-respectful__ classifiers (see Definition 7).
  - Lemma 9: If $\epsilon < d(X_+,X_-)/4$, then set of $\epsilon$__-respectful__ classifiers is guaranteed to be non-empty.
  - Theorem 13 ((non)-linear classifier, random augment): For $r < \epsilon$, worst-case margin is $\geq \frac{1}{2\sqrt{2}} \sqrt{\frac{log(N/d)}{d}}r$ where $N$ is number of augmentation per samples.
  - Theorem 7: For linear classifier, $r$ can be chosen to be O(max-margin) so worst-case margin is also linear to  max-margin.
- :+1: &nbsp; Franceschi et al., **Robustness of classifiers to uniform $\ell_p$ and Gaussian noise**, 2018.
- :+1: &nbsp; Fawsi et al., **Robustness of classifiers: from adversarial to random noise**, NeurIPS 2016.
- :+1: &nbsp; Fawsi et al., **The robustness of deep networks: A geometrical perspective**, 2017.
- Tramer and Boneh, **Adversarial Training and Robustness for Multiple Perturbations**, NeurIPS 2019. 
  - Analyze the same simple dataset as Tsipras et al. 2019 (1 robust and d non-robust features) and show that the robustness against $\ell_\infty$ and $\ell_1$ perturbations is "mutually exclusive" (adversarial accuracy in one comes at a cost of the other) with a specific perturbation size. A more complicated proof shows a similar (but tighter) relationship between $\ell_\infty$ and rotation-translation (RT) perturbation.
  - They analyze that "affine transformation" of multiple perturbation types (actually more like convex combination of points in each perturbation set) and show that for linear classifiers, affine transformation of $\ell_p$-based attacks is equivalent to their union. But this is not the case for "affine transformation" between $\ell_\infty$ and RT (note that affine transformation with RT does not follow the strict definition) as well as in the nonlinear case. 
  - On MNIST and CIFAR-10, they show adversarial training on two types of perturbations by training the model on (1) both samples and take average of the loss (avg) or (2) only the stronger one (max). The two methods have comparable robustness. Both perform better than single-perturbation models on the perturbation they are not trained on but worse on the one they are trained on.
  - They also confirm an observation that models trained on $\ell_\infty$ "learn a threshold function in the first layer" on MNIST which results in gradient masking for $\ell_1$ and $\ell_2$ attacks. But, perhaps interestingly, models trained on $\ell_1$ or $\ell_2$ perturbation directly do not learn this "trick".
  - They also propose a more efficient $$\ell_1$$-attack than PGD and less expensive than EAD attack.


### Hardness of Defense
- A. Fawzi, O. Fawzi, and P. Frossard, **Analysis of classifiers’ robustness to adversarial perturbations.**
- Schmidt et al., **Adversarially Robust Generalization Requires More Data**, NeurIPS 2018.
  - Study two simple data distributions (Gaussian and Bernoulli) and size of training set required for __any__ learning algorithm to have robust generalization. The gap between robust and normal generalization is a factor of $O(\sqrt{d})$ for Gaussian. For Bernoulli, with thresholding, the gap is $O(1)$.
  - Experiment on MNIST shows that more training data is more robust, normal < threshold < adversarial training < adversarial training + threshold.
- Tsipras et al., **Robustness May Be at Odds with Accuracy**, ICLR 2019.
- :+1: &nbsp; Gilmer et al., **Motivating the Rules of the Game for Adversarial Example Research**, 2018. [[link]](https://arxiv.org/abs/1807.06732)
  - Analyze and propose different threat models for adversarial examples in the context of real-world security-critical applications
- :+1: &nbsp; Mahloujifar et al., **The Curse of Concentration in Robust Learning: Evasion and Poisoning Attacks from Concentration of Measure**, 2018.
  - Prove existence of adversarial examples with an assumption on Concentration of the measure. Shows that normal Levy family satisfies this assumption.
- :+1: &nbsp; Shafahi et al., **Are adversarial examples inevitable?**, ICLR 2019.
  - Provide bound on probability of finding an adversarial examples, independent on classifiers but dependent on data distribution (shape of input domain and __density__). The proofs are shown for a unit ball and a unit cube with upper-bounded density. Lower density and higher dimension mean higher probability.
  - However, high dimension alone does not account for more susceptibility because the density also implicitly depends on dimension in an exponential manner which can end up nullifying an increase in dimension (that does not affect intrinsic density i.e. image resizing). Note that $\epsilon$ does increase for higher-dimension MNIST, but it scales with $\sqrt{n}$ for $\ell_2$. See Figure 4b. The authors argue that the probability rather depends on the density (or data __complexity__), comparing between two datasets of equal dimension.
  - Probability of finding adversarial examples within $\epsilon$-ball is __at least__ $1 - U_c e^{-\pi \epsilon^2/2\pi}$ where $U_c$ is an upperbound of density function of class $c$.
- Mahloujifar et al., **Empirically Measuring Concentration: Fundamental Limits on Intrinsic Robustness**, 2019.
- :+1: &nbsp; Dohmatob, **Limitations of adversarial robustness: strong No Free Lunch Theorem**, ICML 2019.
  - Provide a generalization of other previous works (Gilmer et al. 2018, Tsipras et al. 2018, etc.). Covers both geodesic distance and $\ell_p$-norm on flat space. Main tools are borrowed from measure theory (Talagrand W2 transportation-cost inequality, blow-up property).

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
