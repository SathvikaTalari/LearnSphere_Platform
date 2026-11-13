"""
Battle & Compete Engine for LearnSphere.
Provides curated quiz questions, fallback generation, benchmark leaderboard,
and game state helpers for AI Battle Arena.
"""

import random

BATTLE_QUESTIONS = {
    "Machine Learning": {
        "easy": [
            {
                "question": "Which of the following is an example of supervised machine learning?",
                "options": ["A) Predicting housing prices using past sales data", "B) Grouping unlabeled customer records by purchase history", "C) Finding anomalies in server network logs with no labels", "D) Reducing 100 features down to 2 principal components"],
                "answer": "A",
                "explanation": "Predicting house prices uses labeled historical examples (features + actual sale prices), making it a classic supervised regression problem.",
                "points": 10
            },
            {
                "question": "What is the primary difference between classification and regression?",
                "options": ["A) Classification predicts categories; regression predicts continuous numbers", "B) Regression is unsupervised; classification is supervised", "C) Classification cannot use decision trees", "D) Regression only works on image datasets"],
                "answer": "A",
                "explanation": "Classification outputs discrete categorical class labels (e.g. Spam/Ham), while regression predicts continuous numeric values (e.g. price, temperature).",
                "points": 10
            },
            {
                "question": "What does a model suffer from when it achieves 100% accuracy on training data but 52% on test data?",
                "options": ["A) Overfitting (High Variance)", "B) Underfitting (High Bias)", "C) High entropy decay", "D) Premature convergence"],
                "answer": "A",
                "explanation": "A vast performance gap between training accuracy and test accuracy is the hallmark of overfitting: the model memorized training noise.",
                "points": 10
            },
            {
                "question": "Which metric evaluates a binary classifier when positive cases are very rare (e.g., 0.1% fraud)?",
                "options": ["A) Precision, Recall, and PR-AUC", "B) Raw Accuracy", "C) Mean Squared Error", "D) R-squared Score"],
                "answer": "A",
                "explanation": "Raw accuracy is deceptive in imbalanced sets because predicting 'No Fraud' 100% of the time yields 99.9% accuracy. Precision and Recall focus on the true positive detections.",
                "points": 10
            },
            {
                "question": "What does K-Fold Cross-Validation accomplish?",
                "options": ["A) Tests model generalization across multiple held-out partitions of data", "B) Reduces the number of columns in your dataset", "C) Automatically chooses the optimal neural network depth", "D) Eliminates the need for training labels"],
                "answer": "A",
                "explanation": "K-Fold splits data into K equal partitions, training on K-1 and evaluating on 1, repeating K times to ensure every sample is tested.",
                "points": 10
            }
        ],
        "medium": [
            {
                "question": "In the Bias-Variance tradeoff, what typically happens when you increase the depth of a Decision Tree?",
                "options": ["A) Bias decreases while variance increases", "B) Bias increases while variance decreases", "C) Both bias and variance decrease to zero", "D) Neither bias nor variance change"],
                "answer": "A",
                "explanation": "Deeper trees can model increasingly complex decision boundaries, which lowers bias but increases variance and susceptibility to overfitting.",
                "points": 20
            },
            {
                "question": "How does L1 Regularization (Lasso) differ mathematically from L2 Regularization (Ridge)?",
                "options": ["A) L1 adds the sum of absolute weight values; L2 adds the sum of squared weights", "B) L1 multiplies weights by learning rate; L2 divides weights by batch size", "C) L1 is only used in neural networks; L2 is only used in linear models", "D) L1 forces all weights to be positive numbers"],
                "answer": "A",
                "explanation": "L1 penalty is λΣ|w| (inducing sparse weights equal to zero), while L2 penalty is λΣw² (shrinking weights smoothly toward zero).",
                "points": 20
            },
            {
                "question": "In Random Forests, what two mechanisms ensure that individual trees remain de-correlated?",
                "options": ["A) Bootstrap data sampling (bagging) and random feature subsets at each split", "B) Gradient boosting and AdaBoost sample reweighting", "C) Early stopping and learning rate annealing", "D) L1 penalty and weight clipping"],
                "answer": "A",
                "explanation": "Random Forests combine bootstrap aggregating (bagging) with random feature subspace selection at each split so the trees make independent errors.",
                "points": 20
            },
            {
                "question": "What happens if you run K-Means Clustering on unstandardized data where one feature ranges from 0 to 100,000 and another from 0 to 1?",
                "options": ["A) The large-magnitude feature will completely dominate Euclidean distance calculations", "B) The algorithm will run 10 times faster", "C) The centroids will automatically normalize themselves", "D) The algorithm will throw a division-by-zero error"],
                "answer": "A",
                "explanation": "Because K-Means calculates Euclidean distance, features with large scales dominate differences. Preprocessing with StandardScaler is mandatory.",
                "points": 20
            },
            {
                "question": "What is the key advantage of an ROC-AUC curve over raw classification accuracy?",
                "options": ["A) It evaluates performance across all possible classification probability thresholds", "B) It does not require any ground truth labels", "C) It only works when accuracy is 100%", "D) It eliminates the need for a validation set"],
                "answer": "A",
                "explanation": "ROC-AUC plots True Positive Rate vs False Positive Rate across every possible threshold, giving a threshold-independent measure of discriminative power.",
                "points": 20
            }
        ],
        "hard": [
            {
                "question": "What is the primary difference between Bagging (e.g. Random Forest) and Boosting (e.g. XGBoost)?",
                "options": ["A) Bagging trains independent trees in parallel; Boosting trains trees sequentially to correct past errors", "B) Bagging increases bias; Boosting increases variance", "C) Bagging only works on regression; Boosting only works on classification", "D) Boosting averages predictions while Bagging takes the maximum"],
                "answer": "A",
                "explanation": "Bagging fits trees in parallel on bootstrap samples to reduce variance. Boosting fits trees sequentially, with each new tree trained on the pseudo-residuals of the previous ensemble to reduce bias.",
                "points": 30
            },
            {
                "question": "In Support Vector Machines, what mathematical property enables the 'Kernel Trick' to work without computing high-dimensional coordinates?",
                "options": ["A) The optimization dual problem depends only on the inner dot product between sample vectors", "B) The margin is always perpendicular to the eigenvectors", "C) Slack variables cancel out non-linear terms", "D) The Gram matrix is strictly lower triangular"],
                "answer": "A",
                "explanation": "Because the SVM dual objective depends solely on inner products x_i · x_j, replacing the dot product with a kernel function K(x_i, x_j) computes distances in Hilbert space implicitly.",
                "points": 30
            },
            {
                "question": "Which issue is caused by the 'Curse of Dimensionality' in high-dimensional feature spaces?",
                "options": ["A) All data points become approximately equidistant from one another in Euclidean space", "B) Gradient descent steps become infinitely large", "C) The model automatically underfits the training set", "D) Memory consumption drops to zero"],
                "answer": "A",
                "explanation": "In high dimensions, the ratio of distance between the nearest and farthest neighbor approaches 1, meaning distance-based algorithms like KNN and K-Means lose discriminative power.",
                "points": 30
            }
        ]
    },

    "Deep Learning": {
        "easy": [
            {
                "question": "What is the main role of an activation function in a neural network layer?",
                "options": ["A) To introduce non-linearity so the network can learn complex patterns", "B) To reset the model weights to zero after every epoch", "C) To convert image pixels from RGB to grayscale", "D) To increase the GPU clock speed"],
                "answer": "A",
                "explanation": "Without non-linear activations, stacking multiple linear dense layers is mathematically equivalent to a single linear matrix transformation.",
                "points": 10
            },
            {
                "question": "What does a single 'epoch' mean in deep learning model training?",
                "options": ["A) One complete forward and backward pass through the entire training dataset", "B) Processing one single image or batch", "C) Training until loss reaches zero", "D) The time it takes to save model weights to disk"],
                "answer": "A",
                "explanation": "An epoch represents one full iteration where the algorithm has observed every training sample in the dataset once.",
                "points": 10
            },
            {
                "question": "Which loss function is standard for training multi-class classification neural networks?",
                "options": ["A) Categorical Cross-Entropy Loss", "B) Mean Squared Error", "C) Hinge Loss", "D) Cosine Distance"],
                "answer": "A",
                "explanation": "Categorical Cross-Entropy paired with a Softmax output layer measures the divergence between predicted probability distributions and true one-hot labels.",
                "points": 10
            }
        ],
        "medium": [
            {
                "question": "What is the Vanishing Gradient problem and which activation function largely resolved it?",
                "options": ["A) Gradients shrink exponentially in deep layers; ReLU resolved it", "B) Weights explode to infinity; Sigmoid resolved it", "C) Loss becomes negative; Softmax resolved it", "D) Learning rate decays to zero; Tanh resolved it"],
                "answer": "A",
                "explanation": "Sigmoid and Tanh have derivatives < 1 that multiply through the chain rule, shrinking gradients in early layers. ReLU has a constant derivative of 1 for positive inputs, eliminating vanishing gradients.",
                "points": 20
            },
            {
                "question": "What does Dropout do during the training phase of a neural network?",
                "options": ["A) Randomly zeroes out a fraction of neuron activations to prevent co-adaptation", "B) Permanently deletes the smallest weights from the model", "C) Skips the backward pass on 50% of the training epochs", "D) Drops training samples that have high loss"],
                "answer": "A",
                "explanation": "Dropout randomly deactivates a fraction p of neurons during each training step, forcing the network to learn redundant, robust representations rather than co-adapting.",
                "points": 20
            },
            {
                "question": "How does Batch Normalization accelerate neural network training?",
                "options": ["A) Normalizes layer inputs across mini-batches, stabilizing internal covariate shift", "B) Decreases the number of layers in the model", "C) Eliminates the backward propagation pass", "D) Converts 32-bit floating point numbers to 8-bit integers"],
                "answer": "A",
                "explanation": "Batch Normalization standardizes each layer's inputs to have zero mean and unit variance per mini-batch, smoothing the optimization landscape and permitting higher learning rates.",
                "points": 20
            }
        ],
        "hard": [
            {
                "question": "Why do Residual Networks (ResNets) use skip connections (identity shortcuts)?",
                "options": ["A) They allow gradients to flow directly backward without attenuation: ∂L/∂x = ∂L/∂F + ∂L/∂x", "B) They halve the number of parameters in the convolutional kernels", "C) They convert convolutional layers into recurrent layers", "D) They prevent images from needing data augmentation"],
                "answer": "A",
                "explanation": "Skip connections formulate the layer as F(x) + x. In backpropagation, the gradient includes a constant +1 term, allowing gradients to propagate directly to early layers without vanishing.",
                "points": 30
            },
            {
                "question": "In the Adam optimizer, what two moving statistics are tracked for each parameter?",
                "options": ["A) First moment (mean of gradients) and Second raw moment (uncentered variance of gradients)", "B) Hessian diagonal and Fisher Information Matrix", "C) Minimum and Maximum observed weights", "D) Training loss and Validation loss"],
                "answer": "A",
                "explanation": "Adam combines Momentum (exponential moving average of gradients: m_t) and RMSprop (exponential moving average of squared gradients: v_t) with bias correction.",
                "points": 30
            }
        ]
    },

    "Neural Networks": {
        "easy": [
            {
                "question": "What are the three core layers of an artificial neural network?",
                "options": ["A) Input Layer, Hidden Layer(s), Output Layer", "B) Data Layer, Processing Layer, SQL Layer", "C) Kernel Layer, Filter Layer, Matrix Layer", "D) Training Layer, Validation Layer, Testing Layer"],
                "answer": "A",
                "explanation": "Standard neural networks receive signals via the Input Layer, transform them through one or more Hidden Layers, and output predictions via the Output Layer.",
                "points": 10
            },
            {
                "question": "Which algorithm computes the gradients needed to update neural network weights?",
                "options": ["A) Backpropagation", "B) K-Means", "C) Principal Component Analysis", "D) Dijkstra's Algorithm"],
                "answer": "A",
                "explanation": "Backpropagation uses the calculus chain rule to compute the gradient of the loss function with respect to every weight in the network.",
                "points": 10
            }
        ],
        "medium": [
            {
                "question": "What is the mathematical output of the Softmax activation function applied to a vector of logits?",
                "options": ["A) A probability distribution where values are between 0 and 1 and sum to 1.0", "B) Binary 0 or 1 integers", "C) Negative logarithmic odds", "D) A single scalar float between -1.0 and +1.0"],
                "answer": "A",
                "explanation": "Softmax exponentiates logits and divides by the sum of exponentials, creating a calibrated multi-class probability distribution summing to 1.0.",
                "points": 20
            },
            {
                "question": "What is the 'Dying ReLU' problem?",
                "options": ["A) When neurons output zero for all inputs and their gradient permanently stays zero", "B) When weights grow so large that memory overflows", "C) When the loss oscillates and diverges", "D) When GPU temperatures exceed threshold limits"],
                "answer": "A",
                "explanation": "If a large negative gradient knocks a ReLU neuron into negative territory for all data, its gradient becomes 0 everywhere, permanently freezing its weights.",
                "points": 20
            }
        ],
        "hard": [
            {
                "question": "What does the Universal Approximation Theorem state regarding Multi-Layer Perceptrons?",
                "options": ["A) A feedforward network with a single hidden layer and non-linear activation can approximate any continuous function on a compact subset of ℝⁿ", "B) A network with enough layers will never overfit", "C) Any model can be solved analytically without gradient descent", "D) All neural networks converge to the global minimum in polynomial time"],
                "answer": "A",
                "explanation": "The theorem guarantees representational capacity: given sufficient hidden units and non-linear activations, an MLP can approximate any continuous function arbitrarily well.",
                "points": 30
            }
        ]
    },

    "Computer Vision": {
        "easy": [
            {
                "question": "What operation does a 2D Convolutional layer perform on an input image?",
                "options": ["A) Slides a small filter matrix across pixels to compute local element-wise products and sums", "B) Deletes random pixels from the image borders", "C) Transposes the RGB channels into grayscale vectors", "D) Sorts pixel intensities in ascending order"],
                "answer": "A",
                "explanation": "A convolutional kernel slides across the image tensor, calculating the dot product between kernel weights and local pixel patches to extract feature maps.",
                "points": 10
            },
            {
                "question": "What is the primary purpose of a Max Pooling layer in a CNN?",
                "options": ["A) Reduces spatial dimensions (height and width) to control compute and introduce translation invariance", "B) Increases image resolution by 2x", "C) Normalizes color saturation across RGB channels", "D) Calculates the cross-entropy loss"],
                "answer": "A",
                "explanation": "Max pooling downsamples feature maps by taking the maximum value in local windows (e.g. 2x2), shrinking spatial size while retaining dominant activations.",
                "points": 10
            }
        ],
        "medium": [
            {
                "question": "Why do modern CNN architectures use multiple stacked 3x3 filters instead of a single 7x7 filter?",
                "options": ["A) Stacked 3x3 layers have fewer parameters while introducing more non-linear activations", "B) 7x7 filters cannot detect diagonal edges", "C) GPUs cannot process numbers larger than 3", "D) 3x3 filters eliminate the need for training labels"],
                "answer": "A",
                "explanation": "Three 3x3 convolutions cover the same 7x7 receptive field but use 3*(3²*C²)=27C² parameters instead of 49C², while adding three non-linear ReLU stages instead of one.",
                "points": 20
            },
            {
                "question": "What is Data Augmentation in Computer Vision, and why is it beneficial?",
                "options": ["A) Applying rotations, flips, and crops to artificially expand dataset diversity and reduce overfitting", "B) Increasing image brightness to train faster", "C) Duplicating exact identical images 10 times", "D) Compressing images with JPEG algorithms"],
                "answer": "A",
                "explanation": "Augmentation introduces realistic geometric and photometric distortions, training the model to be invariant to angle, scale, and lighting changes.",
                "points": 20
            }
        ],
        "hard": [
            {
                "question": "In object detection architectures like YOLO, what is the role of Non-Maximum Suppression (NMS)?",
                "options": ["A) Eliminates redundant overlapping bounding boxes for the same object, keeping only the highest-confidence detection", "B) Converts bounding box coordinates from normalized floats to integers", "C) Increases the frame rate of video playback", "D) Suppresses dark pixels before feeding into the backbone"],
                "answer": "A",
                "explanation": "NMS calculates Intersection-over-Union (IoU) between candidate boxes and prunes boxes with high overlap that have lower confidence scores than the primary detection.",
                "points": 30
            }
        ]
    },

    "NLP": {
        "easy": [
            {
                "question": "What is 'Tokenization' in Natural Language Processing?",
                "options": ["A) Breaking raw text strings into discrete units (words, subwords, or characters)", "B) Encrypting text with private cryptographic keys", "C) Translating English sentences into German", "D) Removing capital letters from a text file"],
                "answer": "A",
                "explanation": "Tokenization converts continuous text into numerical token IDs that can be mapped into embeddings for neural models.",
                "points": 10
            },
            {
                "question": "What do Word Embeddings (like Word2Vec or GloVe) capture about words?",
                "options": ["A) Semantic relationships and contextual similarity mapped as dense vectors in continuous vector space", "B) The alphabetical order of characters in the dictionary", "C) The exact file size of the word on disk", "D) The number of syllables in the word"],
                "answer": "A",
                "explanation": "Embeddings place words with similar meanings close together in vector space, capturing semantic analogies like King - Man + Woman = Queen.",
                "points": 10
            }
        ],
        "medium": [
            {
                "question": "What major breakthrough did the Self-Attention mechanism in Transformers bring over Recurrent Neural Networks (RNNs)?",
                "options": ["A) Allowed full parallel processing of all tokens simultaneously, removing sequential recurrence bottlenecks", "B) Eliminated the need for word embeddings", "C) Reduced parameter counts to under 1,000", "D) Allowed models to train without backpropagation"],
                "answer": "A",
                "explanation": "RNNs must process words one-by-one sequentially. Self-attention processes all tokens concurrently in parallel, allowing massive GPU scaling and long-range dependencies.",
                "points": 20
            },
            {
                "question": "Why are Positional Encodings injected into Transformer token embeddings?",
                "options": ["A) Self-Attention is permutation-invariant and has no inherent sense of word order", "B) To make sentences grammatically correct", "C) To convert uppercase characters into lowercase", "D) To prevent the model from hallucinating"],
                "answer": "A",
                "explanation": "Because self-attention computes dot products across all pairs symmetrically, 'Dog bites man' and 'Man bites dog' would produce identical sets of representations without positional encodings.",
                "points": 20
            }
        ],
        "hard": [
            {
                "question": "What is the difference between BERT and GPT in terms of Transformer architecture and training objective?",
                "options": ["A) BERT is an encoder trained on masked tokens bidirectionally; GPT is an autoregressive decoder trained on causal next-token prediction", "B) BERT is an image model; GPT is a text model", "C) BERT uses recurrent cells; GPT uses convolutional kernels", "D) BERT does not use self-attention"],
                "answer": "A",
                "explanation": "BERT uses bidirectional Transformer encoders trained on Masked Language Modeling (filling blanks). GPT uses causal Transformer decoders masked to attend only to past tokens for generation.",
                "points": 30
            }
        ]
    },

    "Statistics": {
        "easy": [
            {
                "question": "What is the Median of a dataset?",
                "options": ["A) The middle value when all numbers are sorted in ascending order", "B) The sum of all numbers divided by the count", "C) The most frequently occurring number", "D) The difference between maximum and minimum"],
                "answer": "A",
                "explanation": "The median is the 50th percentile value in a sorted list. Unlike the mean, it is robust against extreme outliers.",
                "points": 10
            },
            {
                "question": "If two events A and B are mutually exclusive, what is P(A and B)?",
                "options": ["A) 0", "B) 1", "C) P(A) * P(B)", "D) 0.5"],
                "answer": "A",
                "explanation": "Mutually exclusive events cannot occur simultaneously by definition, so the probability of both occurring together is strictly 0.",
                "points": 10
            }
        ],
        "medium": [
            {
                "question": "What does the Central Limit Theorem state?",
                "options": ["A) The distribution of sample means approaches a normal distribution as sample size grows, regardless of population distribution", "B) All machine learning algorithms converge to 100% accuracy with infinite data", "C) The variance of a dataset always equals its mean", "D) Extreme outliers become impossible in large datasets"],
                "answer": "A",
                "explanation": "The CLT proves that adding independent random variables produces a bell-shaped Gaussian distribution in the limit, underpinning statistical inference and hypothesis testing.",
                "points": 20
            },
            {
                "question": "What is the difference between Type I Error and Type II Error in hypothesis testing?",
                "options": ["A) Type I is a False Positive (rejecting a true null); Type II is a False Negative (failing to reject a false null)", "B) Type I is a calculation mistake; Type II is data corruption", "C) Type I happens in small datasets; Type II happens in large datasets", "D) Type I error is always 0.05"],
                "answer": "A",
                "explanation": "Type I error (α) is convicting an innocent person (false positive). Type II error (β) is acquitting a guilty person (false negative).",
                "points": 20
            }
        ],
        "hard": [
            {
                "question": "In Bayesian inference, what does the Posterior probability represent according to Bayes' Theorem?",
                "options": ["A) The updated probability of a hypothesis given observed evidence: P(H|E) = [P(E|H) * P(H)] / P(E)", "B) The initial unverified assumption before seeing data", "C) The probability of making a Type I error", "D) The ratio of variance to standard deviation"],
                "answer": "A",
                "explanation": "Bayes' theorem updates prior belief P(H) via likelihood P(E|H) divided by marginal evidence P(E) to compute the updated posterior distribution P(H|E).",
                "points": 30
            }
        ]
    }
}

# Alias mapping for battle topics
TOPIC_MAP = {
    "machine learning": "Machine Learning",
    "ml": "Machine Learning",
    "deep learning": "Deep Learning",
    "dl": "Deep Learning",
    "neural networks": "Neural Networks",
    "nn": "Neural Networks",
    "computer vision": "Computer Vision",
    "cv": "Computer Vision",
    "nlp": "NLP",
    "natural language processing": "NLP",
    "statistics": "Statistics",
    "math": "Statistics",
    "python ml": "Machine Learning"
}

def get_fallback_battle_question(topic: str = "Machine Learning", difficulty: str = "medium", q_num: int = 1, asked_questions: list = None) -> dict:
    """Select a high-quality curated battle question that has not been asked in this session."""
    t_clean = TOPIC_MAP.get(topic.lower().strip(), "Machine Learning")
    d_clean = difficulty.lower().strip()
    if d_clean not in ["easy", "medium", "hard"]:
        d_clean = "medium"

    topic_pool = BATTLE_QUESTIONS.get(t_clean, BATTLE_QUESTIONS["Machine Learning"])
    diff_pool = topic_pool.get(d_clean, topic_pool.get("medium", []))

    asked_set = set(q.lower().strip() for q in (asked_questions or []))
    available = [q for q in diff_pool if q["question"].lower().strip() not in asked_set]

    # If all in this difficulty were asked, fallback to any unasked in the topic
    if not available:
        for d in ["medium", "hard", "easy"]:
            cand = [q for q in topic_pool.get(d, []) if q["question"].lower().strip() not in asked_set]
            if cand:
                available = cand
                break

    # If still none available, reset pool
    selected = random.choice(available if available else diff_pool).copy()

    # Ensure points match difficulty
    pts = 10 if d_clean == "easy" else (20 if d_clean == "medium" else 30)
    selected["points"] = pts
    selected["difficulty"] = d_clean
    return selected


