# Long Short-Term Memory (LSTM)

## The Vanishing Gradient Problem
Long Short-Term Memory networks (LSTMs) solve the vanishing gradient problem that affects standard Recurrent Neural Networks (RNNs) when processing long sequences. In standard RNNs, gradients can become astronomically small as they are backpropagated through time, preventing the network from learning long-range dependencies. LSTMs solve this by introducing a cell state, which acts as a conveyor belt running straight down the entire chain with only minor linear interactions. This allows information to flow along it unchanged, effectively carrying gradients backwards without them vanishing. The flow of information into and out of this cell state is carefully regulated by structures called gates.

## The Forget Gate
The first step in an LSTM is to decide what information we're going to throw away from the cell state. This decision is made by a sigmoid layer called the forget gate. It looks at the previous hidden state and the current input, and outputs a number between 0 and 1 for each number in the cell state. A '1' represents "completely keep this" while a '0' represents "completely get rid of this." This mechanism allows the network to flush irrelevant historical context when a new, distinct sequence begins, freeing up capacity for new information.

## The Input Gate and Output Gate
After deciding what to forget, the LSTM uses an input gate to decide what new information will be stored in the cell state. A sigmoid layer decides which values to update, and a tanh layer creates a vector of new candidate values that could be added to the state. These two are multiplied to update the state. Finally, the output gate determines what the next hidden state should be. This hidden state will be based on the newly updated cell state, but filtered. A sigmoid layer decides what parts of the cell state to output, which is then multiplied by the tanh of the cell state, pushing the values to be between -1 and 1.
