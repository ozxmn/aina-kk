import torch, torch.nn.functional as F

@torch.no_grad()
def token_stats(model, tok, texts, max_len=512, device=None):
    device = device or next(model.parameters()).device
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len, add_special_tokens=True).to(device)
    logits = model(**enc).logits[:, :-1]
    return logits, enc["input_ids"][:, 1:], enc["attention_mask"][:, 1:].float()

@torch.no_grad()
def scores_single(logits, labels, mask, chunk=128):
    B, T = labels.shape
    dev = logits.device
    s_ll = torch.zeros(B, device=dev); s_lr = torch.zeros(B, device=dev); s_ent = torch.zeros(B, device=dev)
    s_dev = torch.zeros(B, device=dev); s_var = torch.zeros(B, device=dev)
    for a in range(0, T, chunk):
        lp = F.log_softmax(logits[:, a:a + chunk].float(), -1)
        lab = labels[:, a:a + chunk].to(dev); m = mask[:, a:a + chunk].to(dev)
        ll_tok = lp.gather(-1, lab.unsqueeze(-1)).squeeze(-1)
        rank = (lp > ll_tok.unsqueeze(-1)).sum(-1).float() + 1
        p = lp.exp()
        mu_tok = (p * lp).sum(-1)
        var_tok = (p * lp.pow(2)).sum(-1) - mu_tok.pow(2)
        del p, lp
        s_ll += (ll_tok * m).sum(1); s_lr += (torch.log(rank) * m).sum(1); s_ent += (-mu_tok * m).sum(1)
        s_dev += ((ll_tok - mu_tok) * m).sum(1); s_var += (var_tok * m).sum(1)
    n = mask.to(dev).sum(1).clamp(min=1)
    ll = s_ll / n; logrank = s_lr / n
    return {"ll": ll, "logrank": logrank, "entropy": s_ent / n, "lrr": -ll / logrank.clamp(min=1e-6),
            "fdg": s_dev / torch.sqrt(s_var.clamp(min=1e-6)), "n_tok": n}

@torch.no_grad()
def binoculars(obs_logits, perf_logits, labels, mask, chunk=128):
    dev = perf_logits.device
    B, T = labels.shape
    s_ppl = torch.zeros(B, device=dev); s_x = torch.zeros(B, device=dev)
    for a in range(0, T, chunk):
        plp = F.log_softmax(perf_logits[:, a:a + chunk].float(), -1)
        op = F.softmax(obs_logits[:, a:a + chunk].to(dev).float(), -1)
        lab = labels[:, a:a + chunk].to(dev); m = mask[:, a:a + chunk].to(dev)
        s_ppl += (-(plp.gather(-1, lab.unsqueeze(-1)).squeeze(-1)) * m).sum(1)
        s_x += (-(op * plp).sum(-1) * m).sum(1)
        del plp, op
    n = mask.to(dev).sum(1).clamp(min=1)
    return (s_ppl / n) / (s_x / n)
