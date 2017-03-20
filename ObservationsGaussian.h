//
//  ObservationsGaussian.h
//  hiddini
//
//  Created by Johan Pauwels on 16/03/2017.
//
//

#ifndef ObservationsGaussian_h
#define ObservationsGaussian_h

#include "MultivariateGaussian.h"
#include <vector>
#include <cmath>

namespace hiddini
{
    template<typename T>
    class ObservationsGaussian
    {
    public:
        typedef T ProbType;
        typedef Eigen::Matrix<T, 1, Eigen::Dynamic> ProbRow;
        typedef typename MultivariateGaussian<T>::ProbColumn ProbColumn;
        typedef typename MultivariateGaussian<T>::ProbMatrix ProbMatrix;
        typedef Eigen::Matrix<Eigen::Index, 1, Eigen::Dynamic> StateSeqType;
        typedef ProbMatrix ObsSeqType;
        typedef std::vector<ProbMatrix> ProbTensor;
        
        ObservationsGaussian(const Eigen::Index in_nStates, const Eigen::Index in_nDimensions)
        : m_nStates(in_nStates), m_nDimensions(in_nDimensions)
        , m_gaussians(m_nStates, MultivariateGaussian<T>(m_nDimensions))
        , m_meanSum(ProbMatrix::Zero(m_nDimensions, m_nStates))
        , m_covarianceSum(m_nStates, ProbMatrix::Zero(m_nDimensions, m_nDimensions))
        {
        }
        
        // @param   in_means [nDimensions x nStates]
        // @param   in_covariances [nStates x [nDimensions x nDimensions]]
        ObservationsGaussian(const ProbMatrix& in_means, const ProbTensor& in_covariances)
        : m_nStates(in_means.cols()), m_nDimensions(in_means.rows())
        , m_gaussians(m_nStates, MultivariateGaussian<T>(m_nDimensions))
        , m_meanSum(ProbMatrix::Zero(m_nDimensions, m_nStates))
        , m_covarianceSum(m_nStates, ProbMatrix::Zero(m_nDimensions, m_nDimensions))
        {
            for (Eigen::Index iState = 0; iState < m_nStates; ++iState)
            {
                m_gaussians[iState] = MultivariateGaussian<T>(in_means.col(iState), in_covariances[iState]);
            }
        }
        
        const ProbMatrix operator()(const ObsSeqType& in_observationSequence) const
        {
            ProbMatrix obsLikelihoods = ProbMatrix::Zero(m_nStates, in_observationSequence.cols());
            for (Eigen::Index iState = 0; iState < m_nStates; ++iState)
            {
                obsLikelihoods.row(iState) = m_gaussians[iState](in_observationSequence);
            }
            return obsLikelihoods;
        }
        
    protected:
        friend class HMM<ObservationsGaussian<T>>;
        
        Eigen::Index getNumStates() const
        {
            return m_nStates;
        }
        
        void reestimateEmissionParameters(const ObsSeqType& in_observationSequence, const ProbMatrix& in_gammas)
        {
            m_meanSum += in_observationSequence * in_gammas.transpose();
            for (Eigen::Index iState = 0; iState < m_nStates; ++iState)
            {
                //Rabiner
                //const ProbMatrix zeroMeanObs = m_gaussians[iState].meanCentredObservations(in_observationSequence);
                //m_covarianceSum[iState] += (zeroMeanObs.array().rowwise() * in_gammas.row(iState).array()).matrix() * zeroMeanObs.transpose();
                m_covarianceSum[iState] += (in_observationSequence.array().rowwise() * in_gammas.row(iState).array()).matrix() * in_observationSequence.transpose();
            }
        }
        
        void saveEmissionParameters(const ProbColumn& in_gammaSum)
        {
            for (Eigen::Index iState = 0; iState < m_nStates; ++iState)
            {
                const ProbColumn newMean = m_meanSum.col(iState) / in_gammaSum[iState];
                const ProbMatrix meanOuterProduct = newMean * newMean.transpose();
                const ProbMatrix newCovariance = m_covarianceSum[iState] / in_gammaSum[iState] - meanOuterProduct + 0.01 * ProbMatrix::Identity(m_nDimensions, m_nDimensions);
                m_gaussians[iState] = MultivariateGaussian<T>(newMean, newCovariance);
            }
            m_meanSum.setZero();
            m_covarianceSum.assign(m_nStates, ProbMatrix::Zero(m_nDimensions, m_nDimensions));
        }
        
        const ObsSeqType generate(const StateSeqType& in_hiddenSequence) const
        {
            const Eigen::Index seqLength = in_hiddenSequence.size();
            ProbMatrix observedSequence(m_nDimensions, seqLength);
            for (Eigen::Index iSeq = 0; iSeq < seqLength; ++iSeq)
            {
                observedSequence.col(iSeq) = m_gaussians[in_hiddenSequence[iSeq]].generate();
            }
            return observedSequence;
        }
        
    private:
        const Eigen::Index m_nStates;
        const Eigen::Index m_nDimensions;
        std::vector<MultivariateGaussian<T>> m_gaussians;
        ProbMatrix m_meanSum; //[m_nDimensions x m_nStates]
        ProbTensor m_covarianceSum; //[nStates x [m_nDimensions x m_nDimensions]]
    };
}

#endif /* ObservationsGaussian_h */
