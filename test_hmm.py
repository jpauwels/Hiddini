from __future__ import division
import numpy as np
import hiddini

## Config
num_states = 2;
num_symbols = 6;
num_observations = 100;
num_iterations = 500;
tolerance = 1e-6;

# trans = np.rand(num_states, num_states);
# trans = bsxfun(@rdivide, trans, sum(trans, 2));
trans = np.array([[0.95,0.05], [0.10,0.90]])
# emis = rand(num_states, num_symbols);
# emis = bsxfun(@rdivide, emis, sum(emis, 2));
emit = np.array([[1/10, 1/10, 1/10, 1/10, 1/10, 1/2], [1/6,  1/6,  1/6,  1/6,  1/6,  1/6]])
# seq = hmmgenerate(nObservations, trans, emis);
init = np.array([[0.5], [0.5]])

obs_seq = np.array([2, 0, 0, 1, 3, 1, 3, 1, 0, 3, 1, 5, 5, 5, 5, 5, 2, 0, 3, 3])

## Object setup
hmm = hiddini.HMMMultinomial(emit, trans, init);

## MAP decoding
# %timeit 
state_seq, log_obs_prob = hmm.decodeMAP(obs_seq)
print(state_seq, log_obs_prob)

## PMAP decoding
# %timeit 
state_seq, log_obs_prob = hmm.decodePMAP(obs_seq)
print(state_seq, log_obs_prob)

## Training
# % seqs = {seq, hmmgenerate(nObservations, trans, emis), hmmgenerate(nObservations, trans, emis)};
# [newTrans, newEmis] = hmmtrain2(seqs, trans, emis, 'Verbose', true);
hmm = hiddini.HMMMultinomial(num_states, num_symbols)
hmm.train(obs_seq, num_iterations, tolerance)

## Evaluation
# %timeit 
log_obs_prob = hmm.evaluate(obs_seq)
print(log_obs_prob)
print('Done')


from seqlearn.hmm import MultinomialHMM
hmm2 = MultinomialHMM()
hmm2.classes_ = np.array([str(x) for x in range(num_states)])
hmm2.intercept_init_ = np.log(np.squeeze(init))
hmm2.intercept_trans_ = np.log(trans)
hmm2.intercept_final_ = np.zeros(num_states)
hmm2.coef_ = np.log(emit)
onehot = np.zeros([len(obs_seq), num_symbols], dtype=int)
onehot[range(len(obs_seq)), obs_seq] = 1
print(hmm2.predict(onehot))
# hmm2.coef_ = np.clip(np.log(chord_probs.T), -99, 0)
# hmm_smoothed_chord_labels = hmm2.predict(np.eye(chord_probs.shape[0]))

hmm3 = hiddini.HMMRaw(trans, init)
obslik = emit[:, obs_seq]
obslik2 = hiddini.ObservationsMultinomial(emit)(obs_seq)
print(hmm3.decodeMAP(obslik2))
