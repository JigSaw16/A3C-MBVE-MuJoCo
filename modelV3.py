import torch
import torch.nn as nn
import torch.distributions as D


class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()
        self.fc_shared = nn.Linear(input_dim, 64)

        self.fc_policy = nn.Linear(64, 64)
        self.fc_policy_output = nn.Linear(64, output_dim)

        self.fc_value = nn.Linear(64, 64)
        self.fc_value_output = nn.Linear(64, 1)

        # Inicjalizacja thread-specific parametrów θ0 i θ0v
        self.theta0 = torch.zeros(1)  # Przykładowa inicjalizacja wartości początkowej
        self.theta0v = torch.zeros(1)  # Przykładowa inicjalizacja wartości początkowej

    def synchronize_parameters(self, global_theta, global_theta_v):
        self.theta0 = global_theta  # Przypisanie wartości globalnego parametru θ do thread-specific parametru θ0
        self.theta0v = global_theta_v  # Przypisanie wartości globalnego parametru θv do thread-specific parametru θ0v

    def sample_action(self, state, th0):
        policy_probs, _ = self.forward(state)
        dist = D.Categorical(policy_probs)
        action = dist.sample()
        return action.item()

    def forward(self, x):
        x_shared = torch.relu(self.fc_shared(x))

        x_policy = torch.relu(self.fc_policy(x_shared))
        policy_logits = self.fc_policy_output(x_policy)
        policy_probs = torch.softmax(policy_logits, dim=-1)

        x_value = torch.relu(self.fc_value(x_shared))
        value = self.fc_value_output(x_value)

        return policy_probs, value


'''
// Assume global shared parameter vectors θ and θv and global shared counter T = 0
// Assume thread-specific parameter vectors θ0 and θ0v
Initialize thread step counter t ← 1
repeat
    Reset gradients: dθ ← 0 and dθv ← 0.
    Synchronize thread-specific parameters θ0 = θ and θ0v = θv
    tstart = t
    Get state st
    repeat
        Perform at according to policy π(at|st; θ0)
        Receive reward rt and new state st+1
        t ← t + 1
        T ← T + 1
    until terminal st or t − tstart == tmax
    R = 0 for terminal st
    or
    R = V(st, θ0v) for non - terminal st // Bootstrap from last state
    for i ∈ {t − 1, . . . , tstart} do
        R ← ri + γR
        Accumulate gradients wrt θ0: dθ ← dθ + ∇θ0 log π(ai|si; θ0)(R − V (si; θ0v))
        Accumulate gradients wrt θ0v: dθv ← dθv + ∂ (R − V (si; θ0v))2/∂θ0v
    end for
    Perform asynchronous update of θ using dθ and of θv using dθv.
until T > Tmax
'''
