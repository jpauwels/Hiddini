//
//  main.cpp
//  test_hiddini
//
//  Created by Johan Pauwels on 16/03/2017.
//
//

#include <iostream>
#include "HMM.h"
#include "ObservationsRaw.h"
#include "ObservationsDiscrete.h"
#include "ObservationsGaussian.h"
#include "ObservationsGMM.h"
#include "ObservationsTemplateCosSim.h"

int main(int argc, const char * argv[]) {
    
    using namespace hiddini;
    typedef HMM<ObservationsDiscrete<double>> HMMDiscrete;
    
    HMMDiscrete::ProbColumn init = (HMMDiscrete::ProbColumn(2) << 0.5, 0.5).finished();
    std::cout << init << std::endl;
    HMMDiscrete::ProbMatrix trans = (HMMDiscrete::ProbMatrix(2, 2) << 0.95, 0.05, 0.10, 0.90).finished();
    std::cout << trans << std::endl;
    HMMDiscrete::ProbMatrix emit = (HMMDiscrete::ProbMatrix(2, 6) << 1./10, 1./10, 1./10, 1./10, 1./10, 1./2, 1./6,  1./6,  1./6,  1./6,  1./6,  1./6).finished();
    std::cout << emit << std::endl;
    
    HMMDiscrete hmm(emit, trans, init);
    
    HMMDiscrete::ObsSeqType obsSeq = (HMMDiscrete::ObsSeqType(20) << 2, 0, 0, 1, 3, 1, 3, 1, 0, 3, 1, 5, 5, 5, 5, 5, 2, 0, 3, 3).finished();
    
    HMMDiscrete::StateSeqType optStateSeq;
    double obsSeqLogProb;
    std::tie(optStateSeq, obsSeqLogProb) = hmm.decodeMAP(obsSeq);
    std::cout << optStateSeq << std::endl;
    std::cout << obsSeqLogProb << std::endl;
    std::tie(optStateSeq, obsSeqLogProb) = hmm.decodePMAP(obsSeq);
    std::cout << optStateSeq << std::endl;
    std::cout << obsSeqLogProb << std::endl;
    std::cout << "Obs sequence prob: " << hmm.evaluate(obsSeq) << std::endl;
    
    hmm = HMMDiscrete(ObservationsDiscrete<double>(2, 6));
    hmm.train(obsSeq, 500, 1e-6);
    std::cout << "Obs sequence prob: " << hmm.evaluate(obsSeq) << std::endl;
    std::tie(optStateSeq, obsSeqLogProb) = hmm.decodeMAP(obsSeq);
    std::cout << optStateSeq << std::endl;
    std::cout << obsSeqLogProb << std::endl;
    
    HMMDiscrete::ObsSeqType observedSeq;
    HMMDiscrete::StateSeqType hiddenStateSeq;
    std::tie(observedSeq, hiddenStateSeq) = hmm.generate(10);
    std::cout << "Observed sequence: " << observedSeq << std::endl;
    std::cout << "Hidden sequence: " << hiddenStateSeq << std::endl;
    
    typedef HMM<ObservationsGaussian<double>> HMMGaussian;
    HMMGaussian hmm2(ObservationsGaussian<double>(2, 3), trans, init);

    
    typedef HMM<ObservationsRaw<double>> HMMRaw;
    HMMRaw hmm3(ObservationsRaw<double>(2), trans, init);
    typedef ObservationsDiscrete<double> ObservationsDiscrete;
    ObservationsDiscrete observer(emit);
    HMMRaw::ProbMatrix p = observer(obsSeq);
    std::cout << p << std::endl;
    HMMRaw::StateSeqType optStateSeq2;
    double obsSeqLogProb2;
    std::tie(optStateSeq2, obsSeqLogProb2) = hmm3.decodeMAP(p);
    std::cout << "State sequence: " << optStateSeq2 << std::endl;
    std::cout << "Sequence probability: " << obsSeqLogProb2 << std::endl;
    
    std::cout << "\n" << std::endl;
    // nStates = 2, nDimensions = 2
    Eigen::MatrixXd templates = (Eigen::MatrixXd(2, 2) << 1, 2, 3, 4).finished();
    ObservationsTemplateCosSim<double> obs(templates);
    // nObservations = 4
    ObservationsTemplateCosSim<double>::ObsSeqType templateObs((Eigen::MatrixXd(2, 4) << 1, 2, 3, 0, 4, 5, 6, 0).finished());
    std::cout << "Observation likelihoods: " << obs(templateObs) << std::endl;
    
    typedef HMM<ObservationsTemplateCosSim<double>> HMMTemplateCosSim;
    HMMTemplateCosSim hmm4(obs, trans, init);
    HMMTemplateCosSim::StateSeqType optStateSeq4;
    double obsSeqLogProb4;
    std::tie(optStateSeq4, obsSeqLogProb4) = hmm4.decodeMAP(templateObs);
    std::cout << "State sequence: " << optStateSeq4 << std::endl;
    std::cout << "Sequence probability: " << obsSeqLogProb4 << std::endl;
    
    std::tie(optStateSeq4, obsSeqLogProb4) = hmm3.decodeMAP(obs(templateObs));
    std::cout << "State sequence: " << optStateSeq4 << std::endl;
    std::cout << "Sequence probability: " << obsSeqLogProb4 << std::endl;
    
    double confidence;
    std::tie(std::ignore, std::ignore, confidence) = hmm4.decodeMAPWithMedianOPC(templateObs);
    std::cout << "Confidence MedianOPC: " << confidence << std::endl;
    std::tie(std::ignore, std::ignore, confidence) = hmm4.decodeWithPPD(templateObs);
    std::cout << "Confidence PPD: " << confidence << std::endl;
    
    return 0;
}
