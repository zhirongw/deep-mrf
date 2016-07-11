# pixel-rnn

Multi-dimensional LSTMs for image generative models.

You must generate the new pixel using all the pixels that you already synthesized, so that the overall image looks coherent.

TODOs and Roadmaps:
1. Image Generation.
-- train on lfw. make sure i get everything right.
2. Rewrite Sequential PixelModel based on Johnson's code: https://github.com/jcjohnson/torch-rnn/blob/master/LSTM.lua.

Friday:
Refine figures; Captions;
Write the final subsection;

Saturday:
Get results;
Think about quantitative evaluation;
Try to finish the paper;
Paper polish discussion with DH

Today:
1. BUG for 1d gmms loss, all other branches.
Paper writing:
boltzmann machine.
long chain, gradient flow.

---
FIGURES and PAPERS:
1. Overview: unrolling the mrf. How?
4. exp: texture(my results and baselines, x1; different mixtures analysis, patch_size  x1, ),
5. exp: sr(2x,3x,4x, set5, set14) results as chao dong x1. Also a Table.
6. exp: i-gen (VAE baseline, GAN baseline, faces) results x1, optional: VAE architecture x1
i-paint results x1.
9 - 11 figures.
----

----
Things that I am not sure:
-2. Whether to have start tokens, and how. A: Zero borders, and shift the pixels.
-3. Patch --> patch loss or just Patch --> pixel loss. A: Seems patch to patch doesn't hurt
-4. How to make the bidirectional idea applicable to arbitrary size. A: Use Gibbs sampling
5. How to evaluate my generative model?
