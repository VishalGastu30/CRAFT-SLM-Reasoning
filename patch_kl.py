import os

fp = '/home/vishal/Projects/CRAFT/src/phase2_rl/craft_rl_loop.py'
with open(fp, 'r') as f:
    content = f.read()

old_kl = '''                # Approximate KL divergence
                kl_div = (policy_logps - ref_logps).mean()
                kl_divs.append(kl_div.item())
                
                # Actor loss: -advantage * policy_logps + kl_beta * kl_div
                actor_loss = -advantage * policy_logps + kl_beta * kl_div'''

new_kl = '''                # Approximate KL divergence (using Schulman's mathematically sound estimator)
                # ratio = exp(ref - policy)
                # kl = ratio - log(ratio) - 1
                ratio = torch.exp(ref_logps - policy_logps)
                kl_div_tensor = ratio - torch.log(ratio) - 1
                kl_div = kl_div_tensor.mean()
                
                # We log the true sample KL for the controller, but use the sound estimator for the loss
                true_sample_kl = (policy_logps - ref_logps).mean()
                kl_divs.append(true_sample_kl.item())
                
                # Actor loss: -advantage * policy_logps + kl_beta * kl_div
                actor_loss = -advantage * policy_logps + kl_beta * kl_div'''

content = content.replace(old_kl, new_kl)
with open(fp, 'w') as f:
    f.write(content)
print("Patched locally")
