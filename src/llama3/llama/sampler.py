# This software was developed by Ivi Chatzi, Nina Corvelo Benz, Eleni Straitouri, Stratis Tsirtsis, and Manuel Gomez Rodriguez.
# If you use this code, please cite the paper "Counterfactual Token Generation in Large Language Models" by the same authors.

import torch

class Sampler:

    def __init__(self, 
                 sampler_type: str,
                 top_p: float = None,
                 top_k: int = None,
                 top_p_intervention: float = 0.9,
                 intervention_seed: int = 42):
        """
        Initialises Sampler, which handles the sampling stage of token selection.

        Args:
            sampler_type: type of sampler, between {"top-p token", "top-p position", "top-k token", "vocabulary"}
            top_p: p value for top-p sampler
            top_k: k value for top-k sampler
        """
        
        self.sample_dict={"top-p position": lambda probs, rng: self.sample_top_p(probs,rng),
                 "top-p token": lambda probs, rng: self.top_p_gumbel_max(probs,rng),
                 "top-k token": lambda probs, rng: self.top_k_gumbel_max(probs,rng),
                 "vocabulary": lambda probs, rng: self.gumbel_max(probs,rng)
        }
        
        assert sampler_type in self.sample_dict, "Invalid sampler. Select a sampler from "+str([i for i in self.sample_dict])+'.'
        
        self.sampler_type = sampler_type
        self.top_p_intervention = top_p_intervention
        self.intervention_seed = intervention_seed 
        self.top_p = None
        self.top_k = None
        self.type = sampler_type

        if sampler_type in {"top-p token","top-p position"}:
            assert top_p is not None, "top_p missing for sampler "+sampler_type
            assert top_p<1 and top_p>0, "top_p must be between 0 and 1"
            self.top_p=top_p
        elif sampler_type=="top-k token":
            assert top_k is not None, "top_k missing for sampler "+sampler_type
            if not isinstance(top_k, int):
                raise TypeError
            self.top_k=top_k


    def sample(self, probs, rng):
        # calls sampling function according to sampler type
        return self.sample_dict[self.sampler_type](probs,rng)
    
    def intervention(self, probs, fixed_token_idx):
        # find top-p tokens
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > self.top_p_intervention
        probs_sort[mask] = 0.0

        # remove token selected from factual generation
        probs_sort[0][(probs_idx[0] == fixed_token_idx).nonzero().flatten()]=0.0
        # if the factual token was the only token in top-p, return next most probable token
        if torch.count_nonzero(probs_sort)==0.0:
            return probs_idx[0][1]
        
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

        # sample from (the rest of) the top-p tokens
        rng_intervention=torch.Generator(device=probs.device)
        rng_intervention.manual_seed(self.intervention_seed)
        next_token = torch.multinomial(probs_sort, num_samples=1, generator=rng_intervention)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token

    def sample_top_p(self, probs, rng):
        """
        Perform top-p (nucleus) sampling on a probability distribution.

        Args:
            probs (torch.Tensor): Probability distribution tensor.
            p (float): Probability threshold for top-p sampling.

        Returns:
            torch.Tensor: Sampled token indices.

        Note:
            Top-p sampling selects the smallest set of tokens whose cumulative probability mass
            exceeds the threshold p. The distribution is renormalized based on the selected tokens.
        """
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > self.top_p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1, generator=rng)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token
    
    def gumbel_max(self, probs, rng):
        """
        Implement the Gumbel-Max SCM.

        Args:
            probs (torch.Tensor): Probability distribution tensor.

        Returns:
            torch.Tensor: Sampled token indices.

        Note:
            Following the SCM, we sample a noise variable U_i~Gumbel(0,1) for each token i,
            where U_i = -ln(-ln(X_i)) and X_i ~ Unif(0,1).
            The next token is the one with the maximum log probs[i] + U_i  
        """
        # Torch does not support sampling directly from gumbel with a generator
        vars_uniform = torch.rand(
            probs.size(), 
            dtype=probs.dtype, 
            layout=probs.layout, 
            device=probs.device,
            generator=rng
        )
        eps = 1e-20
        gumbel_noise = -torch.log(-torch.log(vars_uniform + eps) + eps)
        log_probs = torch.log(probs)

        next_token = torch.argmax(log_probs + gumbel_noise, keepdim=True, dim=-1)
        return next_token

    def top_p_gumbel_max(self, probs, rng):
        """
        Implement the top-p sampling with the Gumbel-Max SCM.

        Args:
            probs (torch.Tensor): Probability distribution tensor.
            p (float): Cumulative probability threshold

        Returns:
            torch.Tensor: Sampled token indices. 
        """
        vars_uniform = torch.rand(
            probs.size(),
            dtype=probs.dtype,
            layout=probs.layout,
            device=probs.device,
            generator=rng
        )
        # If any noise = 0 log returns -inf
        eps = 1e-20
        gumbel_noise = -torch.log(-torch.log(vars_uniform + eps) + eps)

        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > self.top_p

        # normalize probs
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

        # reorder noises based on sorted probs
        gumbel_noise_reordered=torch.gather(gumbel_noise,dim=-1,index=probs_idx)
        gumbel_noise_reordered[mask] = 0.0
        log_probs = torch.log(probs_sort)
        sampling = log_probs + gumbel_noise_reordered

        next_token = torch.argmax(sampling, keepdim=True, dim=-1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token


    def top_k_gumbel_max(self, probs, rng):
        """
        Implement the top-k sampling with the Gumbel-Max SCM.

        Args:
            probs (torch.Tensor): Probability distribution tensor.
            k (int): Number of tokens to sample from.

        Returns:
            torch.Tensor: Sampled token indices. 
        """
        vars_uniform = torch.rand(
            probs.size(),
            dtype=probs.dtype,
            layout=probs.layout,
            device=probs.device,
            generator=rng
        )
        # If any noise = 0, log() returns -inf
        eps = 1e-20
        gumbel_noise = -torch.log(-torch.log(vars_uniform + eps) + eps)

        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sort[:, self.top_k:] = 0.0

        # normalize probs
        # NOTE: maybe we have to skip this
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

        # reorder noises based on sorted probs 
        gumbel_noise_reordered=torch.gather(gumbel_noise,dim=-1,index=probs_idx)
        gumbel_noise_reordered[:, self.top_k:] = 0.0

        log_probs = torch.log(probs_sort)
        sampling = log_probs + gumbel_noise_reordered

        next_token = torch.argmax(sampling, keepdim=True, dim=-1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token