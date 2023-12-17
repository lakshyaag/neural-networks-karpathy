# Preprocessing
## Tokenization
Define the units ($U$) of your model - this can be a character, subword, word, depending on your use case. 
## Training
This includes:

1. Train-test-split
2. Creating batches of size $B$ with $T$ units: $$
U \times 1 \implies B \times [U \times T]
$$
3. $X$ and $Y$ created through next-unit loops (bigram matrix)
4. Token embedding layer created: $T_{N \times E}$ is a token embedding matrix that takes in a unique list of units ($N$) and learns a $E$-dimensional vector for each
5. Positional embedding layer created: $P_{N \times E}$ is a positional embedding matrix that takes in a list of possible positions and learns a $E$-dimensional vector for each. This is necessary because the self-attention mechanism in transformers doesn't have any inherent sense of position or order of the tokens.
6. Final embedding layer: $E_{N \times E} = T + P$

# Encoder layer
## Attention
The self-attention mechanism allows each token to look and learn from a number of other tokens (in theory, infinitely many). Self-attention is the method the Transformer uses to [**bake the “understanding”**](https://jalammar.github.io/illustrated-transformer/#:~:text=Self%2Dattention%20is%20the%20method%20the%20Transformer%20uses%20to%20bake%20the%20%E2%80%9Cunderstanding%E2%80%9D%20of%20other%20relevant%20words%20into%20the%20one%20we%E2%80%99re%20currently%20processing.) of other relevant units into the unit that we are currently processing.

### A single head (Query-Key-Value to Score)
The first step is to generate three vectors from each of the input vectors, which are contained in $E$.

A head size $H$ is chosen as a hyperparameter.

Three  weight layers are defined,
1. Query weights ($W^Q_{E \times H}$)
2. Key weights ($W^K_{E \times H}$)
3. Value weights ($W^V_{E \times H}$)

Then, the individual vectors are created for each input $x$:
- Query: $Q = X \times W^Q$
- Key: $K$
- Value $V$

The second step is to calculate a score for each query and key. So, for example, $x_1$ with a query vector $q_1$ and $k_1$ will get a score vector $S$
$$
S = \begin{bmatrix}
q_{1} \cdot k_{1}^T & q_{1} \cdot k_{2}^T & \dots & q_{1} \cdot k_{h}^T
\end{bmatrix}
$$
Similarly, all $x_E$ will get their own score vector $S_{E} = Q \times K^T$

Then, divide $S$ with the square root of the head size $H$ and pass through a softmax function as:
$$
\text{softmax}\left(\frac{S}{\sqrt{ H }} \right)
$$
This is then multiplied to the value vector $V$, so overall it becomes:
$$
Z =\text{softmax}\left(\frac{Q K^T}{\sqrt{ H }} \right)V
$$
### Multi-headed attention
Now, this can be further refined by adding multiple such heads and allowing them to communicate. This will help the model to learn about relationships between positions even more by expanding the representation subspace, i.e., projecting it to a higher dimension to capture granularity.

Thus, with $k$ number of heads, each head can independently learn the weight matrices $W_0^Q,W_0^K,W_0^V,\dots,W_k^Q,W_k^K,W_k^V$, and produce a set of scores $z_0,\dots,z_k$.

Then, the set of scores is concatenated to create a score matrix $Z$

The output of this layer is then multiplied with another weight matrix $W^O$ to produce the final output matrix
$$
SA = ZW^O
$$
## Residual pathways and LayerNorm
After the self-attention is computed, it passes through an "Add and Normalize" layer which does:
$$
X = \text{LayerNorm}(X+SA(X))
$$

The above is taken directly from the original paper. However, in Andrej Karpathy's video ([YouTube](https://www.youtube.com/watch?v=kCc8FmEb1nY)), he mentions changing the formula to a more recent version:
$$
X = X+SA(LayerNorm_{1}(X))
$$

## Feed-forward layer
The model takes the self-attention for each vector $x$ in the input and passes it through a feed forward layer, which is comprised of:
1. Linear layer: $FF^1_{E \times GE}$
2. ReLU activation
3. Linear layer: $FF^2_{GE \times E}$

After this, the output passes through another add and norm layer:
$$
X = X + FF(LayerNorm_{2}(X))
$$
## Stacking
The block defined above can now be stacked such that the outputs from one block become the inputs for next one, increasing the number of parameters available.

Once done, the output passes into the decoder layers.

# Decoder layer

