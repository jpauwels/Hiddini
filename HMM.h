//
//  HMM.h
//  hiddini
//
//  Created by Johan Pauwels on 03/03/2017.
//
//

#ifndef HMM_h
#define HMM_h

#include <Eigen/Dense>
#include <iostream>
#include <tuple>
#include <vector>
#include <algorithm>

namespace hiddini
{
template<typename ObsT>
class HMM
{
public:
    typedef typename ObsT::ProbType ProbType;
    typedef typename ObsT::ProbRow ProbRow;
    typedef typename ObsT::ProbColumn ProbColumn;
    typedef typename ObsT::ProbMatrix ProbMatrix;
    typedef typename ObsT::StateSeqType StateSeqType;
    typedef typename ObsT::ObsSeqType ObsSeqType;
    
    HMM(const ObsT& in_observationsObject)
    : m_nStates(in_observationsObject.getNumStates())
    , m_obsObject(in_observationsObject)
    , m_transProbs(m_nStates, m_nStates)
    , m_initProbs(m_nStates)
    {
        m_transProbs.setRandom();
        m_transProbs = m_transProbs.cwiseAbs();
        m_transProbs.array().colwise() *= 1 / m_transProbs.rowwise().sum().array();
        m_initProbs.setRandom();
        m_initProbs = m_initProbs.array().abs();
        m_initProbs *= 1 / m_initProbs.sum();
    }
    
    HMM(const ObsT& in_observationsObject, const ProbMatrix& inTransitionProbs, const ProbColumn& inInitialisationProbs = ProbColumn())
    : m_nStates(inTransitionProbs.rows())
    , m_obsObject(in_observationsObject)
    , m_transProbs(inTransitionProbs)
    , m_initProbs(inInitialisationProbs)
    {
        if (m_initProbs.size() == 0)
        {
            m_initProbs = ProbColumn::Constant(m_nStates, 1./m_nStates);
        }
        if (m_transProbs.cols() != m_nStates)
        {
            throw std::length_error("The transition probability matrix should be square");
        }
        if (m_initProbs.size() != m_nStates)
        {
            std::ostringstream errorMessage;
            errorMessage << "The initialisation probabilities should have length " << m_nStates << " instead of " << m_initProbs.size();
            throw std::length_error(errorMessage.str());
        }
        if (m_obsObject.getNumStates() != m_nStates)
        {
            std::ostringstream errorMessage;
            errorMessage << "The number of observation states should be " << m_nStates << " instead of " << m_obsObject.getNumStates();
            throw std::length_error(errorMessage.str());
        }
    }
    
    const ProbType evaluate(const ObsSeqType& in_observationSequence) const
    {
        return forwardAlgorithm(m_obsObject(in_observationSequence));
    }

    const std::tuple<StateSeqType, ProbType> decodeMAP(const ObsSeqType& in_observationSequence) const
    {
        return viterbi(m_obsObject(in_observationSequence), m_transProbs, m_initProbs);
    }
    
    const std::tuple<StateSeqType, ProbType, ProbMatrix> decodeMAPWithLattice(const ObsSeqType& in_observationSequence) const
    {
        return viterbiWithLattice(m_obsObject(in_observationSequence), m_transProbs, m_initProbs);
    }
    
    const std::tuple<StateSeqType, ProbType> decodePMAP(const ObsSeqType& in_observationSequence) const
    {
        ProbMatrix forwardVars;
        ProbMatrix backwardVars;
        ProbRow scalingFactors;
        std::tie(forwardVars, backwardVars, scalingFactors) = forwardBackwardAlgorithm(m_obsObject(in_observationSequence));
        ProbMatrix gamma = forwardVars.cwiseProduct(backwardVars);
        Eigen::Index nObservations = in_observationSequence.cols();
        StateSeqType stateSequence(nObservations);
//        gamma.colwise().maxCoeff(&stateSequence); TODO file Eigen feature request
        for(Eigen::Index i = 0; i < nObservations; ++i)
            gamma.col(i).maxCoeff(&stateSequence[i]);
        ProbRow logScalingFactors = scalingFactors.array().log();
        logScalingFactors = (logScalingFactors.array() < s_logLowerLimit).select(s_logLowerLimit, logScalingFactors);
        ProbType logProb = -logScalingFactors.sum();
        return std::tie(stateSequence, logProb);
    }
    
    const std::tuple<StateSeqType, ProbType, ProbMatrix> decodePMAPWithLattice(const ObsSeqType& in_observationSequence) const
    {
        ProbMatrix forwardVars;
        ProbMatrix backwardVars;
        ProbRow scalingFactors;
        std::tie(forwardVars, backwardVars, scalingFactors) = forwardBackwardAlgorithm(m_obsObject(in_observationSequence));
        ProbMatrix gammas = forwardVars.cwiseProduct(backwardVars);
        Eigen::Index nObservations = in_observationSequence.cols();
        StateSeqType stateSequence(nObservations);
        //        gamma.colwise().maxCoeff(&stateSequence); TODO file Eigen feature request
        for(Eigen::Index i = 0; i < nObservations; ++i)
            gammas.col(i).maxCoeff(&stateSequence[i]);
        ProbRow logScalingFactors = scalingFactors.array().log();
        logScalingFactors = (logScalingFactors.array() < s_logLowerLimit).select(s_logLowerLimit, logScalingFactors);
        ProbType logProb = -logScalingFactors.sum();
        return std::tie(stateSequence, logProb, gammas);
    }
    
    const std::tuple<StateSeqType, ProbType> decodePV(const ObsSeqType& in_observationSequence) const
    {
        // Posterior pass
        ProbMatrix forwardVars;
        ProbMatrix backwardVars;
        ProbRow scalingFactors;
        std::tie(forwardVars, backwardVars, scalingFactors) = forwardBackwardAlgorithm(m_obsObject(in_observationSequence));
        ProbMatrix gammas = forwardVars.cwiseProduct(backwardVars);
        gammas.array().rowwise() /= gammas.array().colwise().sum();
        
        // Viterbi pass
        return viterbi(gammas, (m_transProbs.array() > 0).select(ProbMatrix::Ones(m_nStates, m_nStates), ProbType(0)), (m_initProbs.array() > 0).select(ProbColumn::Ones(m_nStates), ProbType(0)));
        //(m_transProbs > 0).cast<ProbType>()
    }
    
    const std::tuple<StateSeqType, ProbType, ProbMatrix> decodePVWithLattice(const ObsSeqType& in_observationSequence) const
    {
        // Posterior pass
        ProbMatrix forwardVars;
        ProbMatrix backwardVars;
        ProbRow scalingFactors;
        std::tie(forwardVars, backwardVars, scalingFactors) = forwardBackwardAlgorithm(m_obsObject(in_observationSequence));
        ProbMatrix gammas = forwardVars.cwiseProduct(backwardVars);
        gammas.array().rowwise() /= gammas.array().colwise().sum();
        
        // Viterbi pass
        return viterbiWithLattice(gammas, (m_transProbs.array() > 0).select(ProbMatrix::Ones(m_nStates, m_nStates), ProbType(0)), (m_initProbs.array() > 0).select(ProbColumn::Ones(m_nStates), ProbType(0)));
        //(m_transProbs > 0).cast<ProbType>()
    }
    
    const std::tuple<StateSeqType, ProbType> decode(const ObsSeqType& in_observationSequence, const std::string& decoder) const
    {
        if (decoder == "MAP")
        {
            return decodeMAP(in_observationSequence);
        }
        else if (decoder == "PMAP")
        {
            return decodePMAP(in_observationSequence);
        }
        else if (decoder == "PV")
        {
            return decodePV(in_observationSequence);
        }
        else
        {
            throw std::invalid_argument("Unknown decoder '" + decoder + "'");
        }
    }
    
    const std::tuple<StateSeqType, ProbType, ProbMatrix> decodeWithLattice(const ObsSeqType& in_observationSequence, const std::string& decoder) const
    {
        if (decoder == "MAP")
        {
            return decodeMAPWithLattice(in_observationSequence);
        }
        else if (decoder == "PMAP")
        {
            return decodePMAPWithLattice(in_observationSequence);
        }
        else if (decoder == "PV")
        {
            return decodePVWithLattice(in_observationSequence);
        }
        else
        {
            throw std::invalid_argument("Unknown decoder '" + decoder + "'");
        }
    }
    
    const std::tuple<StateSeqType, ProbType, ProbType> decodeWithPPD(const ObsSeqType& in_observationSequence, const std::string& outputDecoder = "MAP", const std::string& additionalDecoder = "PMAP") const
    {
        StateSeqType optStateSeq;
        ProbType obsSeqLogProb;
        std::tie(optStateSeq, obsSeqLogProb) = decode(in_observationSequence, outputDecoder);
        StateSeqType addStateSeq;
        std::tie(addStateSeq, std::ignore) = decode(in_observationSequence, additionalDecoder);
        const ProbType ppd = static_cast<ProbType>((optStateSeq.array() == addStateSeq.array()).count()) / optStateSeq.size();
        return std::tie(optStateSeq, obsSeqLogProb, ppd);
    }
    
    const std::tuple<StateSeqType, ProbType, ProbType> decodeMAPWithMedianOPC(const ObsSeqType& in_observationSequence) const
    {
        StateSeqType optStateSeq;
        ProbType obsSeqLogProb;
        ProbMatrix lattice;
        std::tie(optStateSeq, obsSeqLogProb, lattice) = decodeMAPWithLattice(in_observationSequence);
        const Eigen::Index nObs = optStateSeq.size();
        //const ProbRow optimalPathProbs = lattice(optStateSeq, Eigen::seq(0, nObs)); //CHECK Eigen more optimal way
        ProbRow optimalPathProbs(nObs);
        for (Eigen::Index iObs = 0; iObs < nObs; ++iObs)
        {
            optimalPathProbs[iObs] = lattice(optStateSeq[iObs], iObs);
        }
        const ProbRow optimalPathContributions = optimalPathProbs -  (ProbRow(nObs) << 0, optimalPathProbs.head(nObs-1)).finished();
        const ProbType medianOPC = median(optimalPathContributions);
        return std::tie(optStateSeq, obsSeqLogProb, medianOPC);
    }

    void train(const ObsSeqType& in_observationSequence, const Eigen::Index inMaxIterations, const ProbType inTolerance, const bool verbose=true)
    {
        ProbType prevLogLH = 0;
        bool converged = false;
        for (Eigen::Index i = 0; i < inMaxIterations; ++i)
        {
            ProbMatrix xiSum;
            ProbColumn gammaSum;
            ProbType logLH;
            std::tie(xiSum, gammaSum, m_initProbs, logLH) = baumWelchIteration(in_observationSequence);
            ProbMatrix prevTransProbs = m_transProbs;
            m_transProbs = xiSum.array().colwise() / xiSum.rowwise().sum().array();
            m_obsObject.saveEmissionParameters(gammaSum);
            // Test convergence
            // abs(logLH-prevLogLH) / (1+abs(prevLogLH)) < tolerance && ...
            // norm(this.transProb - prevTransProb, inf) / this.nStates < tolerance %&& ...
            //% norm(guessE - oldGuessE, inf)/size(this.emissionParameters{1},2) < tolerance
            if (std::abs(logLH-prevLogLH) / (1+std::abs(prevLogLH)) < inTolerance)
//                && (m_transProbs - prevTransProbs).lpNorm<Eigen::Infinity>() / m_nStates < inTolerance)
            {
                converged = true;
                break;
            }
            prevLogLH = logLH;
            if (verbose)
            {
                std::cout << "iteration " << i << ": " << logLH << std::endl;
            }
        }
        if (!converged)
        {
            //warning('The training did not converge after %d iterations', maxIterations);
            std::cerr << "The training did not converge after " << inMaxIterations << " iterations";
        }
    }
    
    void train(const std::vector<ObsSeqType>& in_observationSequences, const Eigen::Index inMaxIterations, const ProbType inTolerance, const bool verbose=true)
    {
        ProbType prevLogLH = 1;
        bool converged = false;
        Eigen::Index nSequences = in_observationSequences.size();
        for (Eigen::Index i = 0; i < inMaxIterations; ++i)
        {
            ProbMatrix xiSumSum = ProbMatrix::Zero(m_nStates, m_nStates);
            ProbColumn gammaSumSum = ProbColumn::Zero(m_nStates);
            ProbColumn initProbsSum = ProbColumn::Zero(m_nStates);
            ProbType logLHSum = 0;
            for (Eigen::Index iSequence = 0; iSequence < nSequences; ++iSequence)
            {
                ProbMatrix xiSum;
                ProbColumn gammaSum;
                ProbColumn initProbs;
                ProbType logLH;
                std::tie(xiSum, gammaSum, initProbs, logLH) = baumWelchIteration(in_observationSequences[iSequence]);
                xiSumSum += xiSum;
                gammaSumSum += gammaSum;
                initProbsSum += initProbs;
                logLHSum += logLH;
            }
            m_initProbs = initProbsSum / nSequences;
            ProbMatrix prevTransProbs = m_transProbs;
            m_transProbs = xiSumSum.array().colwise() / xiSumSum.rowwise().sum().array();
            m_obsObject.saveEmissionParameters(gammaSumSum);
            // Test convergence
            // abs(logLH-prevLogLH) / (1+abs(prevLogLH)) < tolerance && ...
            // norm(this.transProb - prevTransProb, inf) / this.nStates < tolerance %&& ...
            //% norm(guessE - oldGuessE, inf)/size(this.emissionParameters{1},2) < tolerance
            if (std::abs(logLHSum-prevLogLH) / (1+std::abs(prevLogLH)) < inTolerance) //TODO check Murphy's convergence criterion
//                && (m_transProbs - prevTransProbs).lpNorm(Eigen::Infinity) / m_nStates < inTolerance)
            {
                converged = true;
                break;
            }
            prevLogLH = logLHSum;
            if (verbose)
            {
                std::cout << "iteration " << i << ": " << logLHSum << std::endl;
            }
        }
        if (!converged)
        {
            std::cerr << "The training did not converge after " << inMaxIterations << " iterations";
        }
    }
    
    const std::tuple<ObsSeqType,StateSeqType> generate(const Eigen::Index in_sequenceLength) const
    {
        ProbColumn initBoundaries(m_nStates);
        initBoundaries[0] = m_initProbs[0];
        ProbMatrix transBoundaries(m_nStates, m_nStates);
        transBoundaries.col(0) = m_transProbs.col(0);
        for (Eigen::Index iCol = 1; iCol < m_nStates; ++iCol)
        {
            initBoundaries[iCol] = initBoundaries[iCol-1] + m_initProbs[iCol];
            transBoundaries.col(iCol) = transBoundaries.col(iCol-1) + m_transProbs.col(iCol);
        }
        ProbRow probs = ProbRow::Random(in_sequenceLength).cwiseAbs();
        StateSeqType hiddenSequence(in_sequenceLength);
        (probs[0] <= initBoundaries.array()).maxCoeff(&hiddenSequence[0]);
        for (Eigen::Index iSeq = 1; iSeq < in_sequenceLength; ++iSeq)
        {
            (probs[iSeq] <= transBoundaries.row(hiddenSequence[iSeq-1]).array()).maxCoeff(&hiddenSequence[iSeq]);
        }
        const ObsSeqType observedSequence = m_obsObject.generate(hiddenSequence);
        return std::tie(observedSequence, hiddenSequence);
    }
    
protected:
    
    typedef Eigen::Matrix<Eigen::Index, Eigen::Dynamic, Eigen::Dynamic> IndexMatrix;
    
    const ProbType forwardAlgorithmUnscaled(const ProbMatrix& in_obsLikelihoods) const
    {
        // Initialisation
        ProbColumn forwardVars = m_initProbs.cwiseProduct(in_obsLikelihoods.col(0));
        // Recursion
        for (Eigen::Index iStep = 1; iStep < in_obsLikelihoods.cols(); ++iStep)
        {
            forwardVars = (m_transProbs.transpose() * forwardVars).cwiseProduct(in_obsLikelihoods.col(iStep));
        }
        // Termination
        return forwardVars.sum();
    }
    
    const ProbType forwardAlgorithm(const ProbMatrix& in_obsLikelihoods) const
    {
        // Initialisation
        ProbColumn forwardVars = m_initProbs.cwiseProduct(in_obsLikelihoods.col(0));
        ProbType logSequenceProb = 0;
        // Recursion
        for (Eigen::Index iStep = 1; iStep < in_obsLikelihoods.cols(); ++iStep)
        {
            forwardVars = (m_transProbs.transpose() * forwardVars).cwiseProduct(in_obsLikelihoods.col(iStep));
            ProbType scalingFactor = 1 / forwardVars.sum();
            forwardVars *= scalingFactor;
            logSequenceProb -= std::log(scalingFactor);
        }
        return logSequenceProb;
    }
    
    const std::tuple<ProbMatrix, ProbMatrix> forwardBackwardAlgorithmUnscaled(const ProbMatrix& in_obsLikelihoods) const
    {
        // Initialisation
        const Eigen::Index nObservations = in_obsLikelihoods.cols();
        ProbMatrix forwardVars(m_nStates, nObservations);
        forwardVars.col(0) = m_initProbs.cwiseProduct(in_obsLikelihoods.col(0));
        ProbMatrix backwardVars(m_nStates, nObservations);
        backwardVars.col(nObservations-1).setOnes();
        // Recursion
        for (Eigen::Index iStep = 1; iStep < in_obsLikelihoods.cols(); ++iStep)
        {
            forwardVars.col(iStep) = (m_transProbs.transpose() * forwardVars.col(iStep-1)).cwiseProduct(in_obsLikelihoods.col(iStep));
        }
        for (Eigen::Index iStep = nObservations-1; iStep > 0; --iStep)
        {
            backwardVars.col(iStep-1) = m_transProbs * in_obsLikelihoods.col(iStep).cwiseProduct(backwardVars.col(iStep));
        }
        // Termination
        return std::tie(forwardVars, backwardVars);
    }
    
    const std::tuple<ProbMatrix, ProbMatrix, ProbColumn> forwardBackwardAlgorithm(const ProbMatrix& in_obsLikelihoods) const
    {
        // Initialisation
        const Eigen::Index nObservations = in_obsLikelihoods.cols();
        ProbMatrix forwardVars(m_nStates, nObservations);
        forwardVars.col(0) = m_initProbs.cwiseProduct(in_obsLikelihoods.col(0));
        ProbRow scalingFactors(nObservations);
        scalingFactors(0) = 1 / forwardVars.col(0).sum();
        forwardVars.col(0) *= scalingFactors(0);
        ProbMatrix backwardVars(m_nStates, nObservations);
        backwardVars.col(nObservations-1).setOnes();
        // Recursion
        for (Eigen::Index iStep = 1; iStep < in_obsLikelihoods.cols(); ++iStep)
        {
            forwardVars.col(iStep) = (m_transProbs.transpose() * forwardVars.col(iStep-1)).cwiseProduct(in_obsLikelihoods.col(iStep));
            scalingFactors(iStep) = 1 / forwardVars.col(iStep).sum();
            forwardVars.col(iStep) *= scalingFactors(iStep);
        }
        for (Eigen::Index iStep = nObservations-1; iStep > 0; --iStep)
        {
            backwardVars.col(iStep-1) = scalingFactors(iStep) * m_transProbs * in_obsLikelihoods.col(iStep).cwiseProduct(backwardVars.col(iStep));
        }
        // Termination
        return std::tie(forwardVars, backwardVars, scalingFactors);
    }
    
    const std::tuple<StateSeqType, ProbType> viterbi(const ProbMatrix& in_obsLikelihoods, const ProbMatrix& in_transProbs, const ProbColumn& in_initProbs) const
    {
        ProbMatrix obsLogLikelihoods = in_obsLikelihoods.array().log();
        obsLogLikelihoods = (obsLogLikelihoods.array() < s_logLowerLimit).select(s_logLowerLimit, obsLogLikelihoods);
        ProbMatrix logTransProbs = in_transProbs.array().log();
        logTransProbs = (logTransProbs.array() < s_logLowerLimit).select(s_logLowerLimit, logTransProbs);
        const Eigen::Index nObservations = in_obsLikelihoods.cols();
        // Initialisation
        ProbColumn logInitProbs = in_initProbs.array().log();
        logInitProbs = (logInitProbs.array() < s_logLowerLimit).select(s_logLowerLimit, logInitProbs);
        ProbColumn bestProbs = logInitProbs.array() + obsLogLikelihoods.col(0).array();
        IndexMatrix prevStates(m_nStates, nObservations);
        // Recursion
        for (Eigen::Index iStep = 1; iStep < nObservations; ++iStep)
        {
            ProbColumn maxAccumulProbs(m_nStates);
//            maxAccumulProbs = (bestProbs + logTransProbs).colwise().maxCoeff(prevStates.col(iStep)); TODO file Eigen request
            for(Eigen::Index i = 0; i < m_nStates; ++i)
                maxAccumulProbs[i] = (bestProbs + logTransProbs.col(i)).maxCoeff(&prevStates(i, iStep));
            bestProbs = maxAccumulProbs + obsLogLikelihoods.col(iStep);
        }
        // Termination
        StateSeqType stateSequence(nObservations);
//        ProbType logProb = bestProbs.maxCoeff(&stateSequence.tail<1>()); OR &stateSequence(Eigen::placeholders::last) OR stateSequence.end()
        // TODO file Eigen request
        ProbType logProb = bestProbs.maxCoeff(&stateSequence(nObservations-1));
        // Backtracking
        for (Eigen::Index iStep = nObservations-1; iStep > 0; --iStep)
        {
            stateSequence(iStep-1) = prevStates(stateSequence(iStep), iStep);
        }
        return std::tie(stateSequence, logProb);
    }
    
    const std::tuple<StateSeqType, ProbType, ProbMatrix> viterbiWithLattice(const ProbMatrix& in_obsLikelihoods, const ProbMatrix& in_transProbs, const ProbColumn& in_initProbs) const
    {
        ProbMatrix obsLogLikelihoods = in_obsLikelihoods.array().log();
        obsLogLikelihoods = (obsLogLikelihoods.array() < s_logLowerLimit).select(s_logLowerLimit, obsLogLikelihoods);
        ProbMatrix logTransProbs = in_transProbs.array().log();
        logTransProbs = (logTransProbs.array() < s_logLowerLimit).select(s_logLowerLimit, logTransProbs);
        const Eigen::Index nObservations = in_obsLikelihoods.cols();
        // Initialisation
        ProbColumn logInitProbs = in_initProbs.array().log();
        logInitProbs = (logInitProbs.array() < s_logLowerLimit).select(s_logLowerLimit, logInitProbs);
        ProbMatrix lattice(m_nStates, nObservations);
        lattice.col(0) = logInitProbs.array() + obsLogLikelihoods.col(0).array();
        IndexMatrix prevStates(m_nStates, nObservations);
        // Recursion
        for (Eigen::Index iStep = 1; iStep < nObservations; ++iStep)
        {
            ProbColumn maxAccumulProbs(m_nStates);
            //            maxAccumulProbs = (lattice.col(i-1) + logTransProbs).colwise().maxCoeff(prevStates.col(iStep)); TODO file Eigen request
            for(Eigen::Index i = 0; i < m_nStates; ++i)
                maxAccumulProbs[i] = (lattice.col(iStep-1) + logTransProbs.col(i)).maxCoeff(&prevStates(i, iStep));
            lattice.col(iStep) = maxAccumulProbs + obsLogLikelihoods.col(iStep);
        }
        // Termination
        StateSeqType stateSequence(nObservations);
        // ProbType logProb = bestProbs.maxCoeff(&stateSequence.tail<1>()); OR &stateSequence(Eigen::placeholders::last) OR stateSequence.end()
        // TODO file Eigen request
        ProbType logProb = lattice.rightCols(Eigen::fix<1>).maxCoeff(&stateSequence(nObservations-1));
        // Backtracking
        for (Eigen::Index iStep = nObservations-1; iStep > 0; --iStep)
        {
            stateSequence(iStep-1) = prevStates(stateSequence(iStep), iStep);
        }
        return std::tie(stateSequence, logProb, lattice);
    }
    
    const std::tuple<ProbMatrix, ProbColumn, ProbColumn, ProbType> baumWelchIteration(const ObsSeqType& in_observationSequence)
    {
        const ProbMatrix obsLikelihoods = m_obsObject(in_observationSequence);
        ProbMatrix forwardVars;
        ProbMatrix backwardVars;
        ProbColumn scalingFactors;
        std::tie(forwardVars, backwardVars, scalingFactors) = forwardBackwardAlgorithm(obsLikelihoods);
        ProbType logLikelihood = -scalingFactors.array().log().sum();
        ProbMatrix xiSum = ProbMatrix::Zero(m_nStates, m_nStates);
        Eigen::Index nObservations = in_observationSequence.cols();
        ProbMatrix gammas = ProbMatrix::Zero(m_nStates, nObservations);
        for (Eigen::Index iStep = 0; iStep < nObservations-1; ++iStep)
        {
            ProbMatrix xi = (scalingFactors(iStep+1) * forwardVars.col(iStep) * obsLikelihoods.col(iStep+1).cwiseProduct(backwardVars.col(iStep+1)).transpose()).cwiseProduct(m_transProbs);
            xiSum += xi;
            gammas.col(iStep) = xi.colwise().sum();
        }
        gammas.col(nObservations-1) = forwardVars.col(nObservations-1);
        const ProbColumn gammaSum = gammas.rowwise().sum();
        const ProbColumn initProbs = gammas.col(0);
        m_obsObject.reestimateEmissionParameters(in_observationSequence, gammas);
        return std::tie(xiSum, gammaSum, initProbs, logLikelihood);
    }
    
    static const ProbType median(const ProbRow& in_vector)
    {
        Eigen::Index nElements = in_vector.size();
        Eigen::Index middleIndex = nElements / 2;
        std::vector<ProbType> vector(in_vector.data(), in_vector.data()+nElements);
        std::nth_element(vector.begin(), vector.begin()+middleIndex, vector.end());
        if (nElements % 2 == 1)
        {
            return vector[middleIndex];
        }
        else
        {
            std::nth_element(vector.begin(), vector.begin()+middleIndex-1, vector.end());
            return 0.5 * (vector[middleIndex] + vector[middleIndex-1]);
        }
    }
    
private:
    static ProbType s_logLowerLimit;
    Eigen::Index m_nStates;
    ObsT m_obsObject;
    ProbMatrix m_transProbs;
    ProbColumn m_initProbs;
};
template<typename ObsT> typename HMM<ObsT>::ProbType HMM<ObsT>::s_logLowerLimit = -999;
}

#endif /* HMM_h */
