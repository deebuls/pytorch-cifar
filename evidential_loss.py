import torch
import torch.nn.functional as F


class EvidentialMSELoss(torch.nn.Module):
    def __init__(self, annealing_step=10000, num_classes=10):
        super(EvidentialMSELoss, self).__init__()

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.annealing_step=annealing_step
        self.num_classes = num_classes
        self.batch_count = 0
        self.shifted_to_full_loss = False

    def one_hot_embedding(self, labels):
        # Convert to One Hot Encoding
        y = torch.eye(self.num_classes, device=self.device)
        return y[labels]

    def relu_evidence(self, y):
        return F.relu(y)

    def celu_evidence(self, y):
        return F.celu(y)


    def exp_evidence(y):
        return torch.exp(torch.clamp(y, -10, 10))


    def softplus_evidence(self, y):
        return F.softplus(y)

    def squareplus_evidence(self, x):
        return (x + torch.sqrt(x**2 + 4))/2


    def kl_divergence(self, alpha):
        beta = torch.ones([1, self.num_classes], dtype=torch.float32, device=self.device)
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)
        lnB = torch.lgamma(S_alpha) - \
            torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        lnB_uni = torch.sum(torch.lgamma(beta), dim=1,
                            keepdim=True) - torch.lgamma(S_beta)

        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)

        kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1,
                      keepdim=True) + lnB + lnB_uni
        return kl


    def loglikelihood_loss(self, y, alpha):
        y = y.to(self.device)
        alpha = alpha.to(self.device)
        S = torch.sum(alpha, dim=1, keepdim=True)
        loglikelihood_err = torch.sum(
            (y - (alpha / S)) ** 2, dim=1, keepdim=True)
        loglikelihood_var = torch.sum(
            alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
        loglikelihood = loglikelihood_err + loglikelihood_var
        return loglikelihood


    def mse_loss(self, y, alpha):
        
        y = y.to(self.device)
        alpha = alpha.to(self.device)
        loglikelihood = self.loglikelihood_loss(y, alpha)

        annealing_coef = torch.min(torch.tensor(
            1.0, dtype=torch.float32, device=self.device), torch.tensor(self.batch_count / self.annealing_step, dtype=torch.float32, device=self.device))

        kl_alpha = (alpha - 1) * (1 - y) + 1
        kl_div = annealing_coef * \
            self.kl_divergence(kl_alpha)
        return loglikelihood + kl_div


    def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device=None):
        y = y.to(device)
        alpha = alpha.to(device)
        S = torch.sum(alpha, dim=1, keepdim=True)

        A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

        annealing_coef = torch.min(torch.tensor(
            1.0, dtype=torch.float32), torch.tensor(epoch_num / annealing_step, dtype=torch.float32))

        kl_alpha = (alpha - 1) * (1 - y) + 1
        kl_div = annealing_coef * \
            self.kl_divergence(kl_alpha, num_classes)
        return A + kl_div


    def forward(self, output, target):
        #self.batch_count += target.shape[0]
        self.batch_count += 1
        if not self.shifted_to_full_loss:
            if (self.batch_count > self.annealing_step):
                print ("Shifted to full loss function")
                self.shifted_to_full_loss = True
        target = self.one_hot_embedding(target.long())
        #evidence = self.relu_evidence(output)
        evidence = self.celu_evidence(output)
              
        alpha = evidence + 1
        loss = torch.mean(self.mse_loss(target, alpha))
        #loss = self.mse_loss(target, alpha)
        return loss


    def edl_log_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
        if not device:
            device = get_device()
        evidence = relu_evidence(output)
        alpha = evidence + 1
        loss = torch.mean(edl_loss(torch.log, target, alpha,
                                  epoch_num, num_classes, annealing_step, device))
        return loss


    def edl_digamma_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
        if not device:
            device = get_device()
        evidence = relu_evidence(output)
        alpha = evidence + 1
        loss = torch.mean(edl_loss(torch.digamma, target, alpha,
                                  epoch_num, num_classes, annealing_step, device))
        return loss

