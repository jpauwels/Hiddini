//
//  ObservationsGMM.h
//  hiddini
//
//  Created by Johan Pauwels on 16/03/2017.
//
//

#ifndef ObservationsGMM_h
#define ObservationsGMM_h

#include "GaussianMixtureModel.h"

namespace hiddini
{
    template<typename T>
    class ObservationsGMM
    {
    public:
        typedef T ProbType;
        typedef typename GaussianMixtureModel<T>::ProbRow ProbRow;
        typedef typename GaussianMixtureModel<T>::ProbColumn ProbColumn;
        typedef typename GaussianMixtureModel<T>::ProbMatrix ProbMatrix;
        typedef typename GaussianMixtureModel<T>::ProbTensor ProbTensor;
        typedef Eigen::Matrix<Eigen::Index, 1, Eigen::Dynamic> StateSeqType;
        typedef ProbMatrix ObsSeqType;
        
        ObservationsGMM(const Eigen::Index in_nStates, const Eigen::Index in_nDimensions, const Eigen::Index in_nComponents)
        : m_nStates(in_nStates), m_nDimensions(in_nDimensions), m_nComponents(in_nComponents)
        , m_gmms(m_nStates, GaussianMixtureModel<T>(m_nDimensions, m_nComponents))
        , m_coefficientSum(m_nStates, ProbRow::Zero(m_nComponents))
        , m_meanSum(m_nStates, ProbMatrix::Zero(m_nDimensions, m_nComponents))
        , m_covarianceSum(m_nStates, ProbTensor(m_nComponents, ProbMatrix::Zero(m_nDimensions, m_nDimensions)))
        {
        }
        
        // @param   in_mixMeans [nStates x [nDimensions x nComponents]]
        // @param   in_mixCoefficients [nStates x [nComponents x [nDimensions x nDimensions]]]
        // @param   in_mixCoefficients [nStates x [1 x nComponents]]
        ObservationsGMM(const std::vector<ProbMatrix>& in_mixMeans, const std::vector<ProbTensor>& in_mixCovariances, const std::vector<ProbRow>& in_mixCoefficients = std::vector<ProbRow>())
        : m_nStates(in_mixMeans.size()), m_nDimensions(in_mixMeans.front().rows()), m_nComponents(in_mixMeans.front().cols())
        , m_gmms(m_nStates, GaussianMixtureModel<T>(m_nDimensions, m_nComponents))
        , m_coefficientSum(m_nStates, ProbRow::Zero(m_nComponents))
        , m_meanSum(m_nStates, ProbMatrix::Zero(m_nDimensions, m_nComponents))
        , m_covarianceSum(m_nStates, ProbTensor(m_nComponents, ProbMatrix::Zero(m_nDimensions, m_nDimensions)))
        {
            if (in_mixCoefficients.size() == 0)
            {
                for (Eigen::Index iState = 0; iState < m_nStates; ++iState)
                {
                    m_gmms[iState] = GaussianMixtureModel<T>(in_mixMeans[iState], in_mixCovariances[iState]);
                }
            }
            else
            {
                for (Eigen::Index iState = 0; iState < m_nStates; ++iState)
                {
                    m_gmms[iState] = GaussianMixtureModel<T>(in_mixMeans[iState], in_mixCovariances[iState], in_mixCoefficients[iState]);
                }
            }
        }
        
        const ProbMatrix operator()(const ObsSeqType& in_observationSequence) const
        {
            ProbMatrix obsLikelihoods = ProbMatrix::Zero(m_nStates, in_observationSequence.cols());
            for (Eigen::Index iState = 0; iState < m_nStates; ++iState)
            {
                obsLikelihoods.row(iState) = m_gmms[iState](in_observationSequence);
            }
            return obsLikelihoods;
        }
        
    protected:
        friend class HMM<ObservationsGMM<T>>;
        
        Eigen::Index getNumStates() const
        {
            return m_nStates;
        }
        
        void reestimateEmissionParameters(const ObsSeqType& in_observationSequence, const ProbMatrix& in_gammas)
        {
            for (Eigen::Index iState = 0; iState < m_nStates; ++iState)
            {
                ProbMatrix componentProbs = m_gmms[iState].posteriorComponentProbs(in_observationSequence); //[nComponents x nTimeSteps]
                ProbMatrix mixGammaCurState = componentProbs.array().colwise() * in_gammas.col(iState).array(); //[nComponents x nTimeSteps]
                ProbColumn mixGammaTotal = mixGammaCurState.rowwise().sum(); //[nComponents x 1]
                m_coefficientSum[iState] += mixGammaTotal.transpose();
                m_meanSum[iState] += in_observationSequence * mixGammaCurState.transpose(); //[nDimensions x nComponents]
                for (Eigen::Index iComponent = 0; iComponent < m_nComponents; ++iComponent)
                {
                    // Rabiner
                    //ProbMatrix zeroMeanObs = m_gmms[iState].meanCentredObservations(in_observationSequence, iComponent); //[nDimensions x nTimeSteps]
                    //m_covarianceSum[iState][iComponent] += (zeroMeanObs.array().rowwise() * mixGammaCurState.row(iComponent).array()).matrix() * zeroMeanObs;//[nDimensions x nDimensions]
                    m_covarianceSum[iState][iComponent] += (in_observationSequence.array().rowwise() * mixGammaCurState.row(iComponent).array()).matrix() * in_observationSequence; //[nDimensions x nDimensions]
                }
            }
        }
        
        void saveEmissionParameters(const ProbColumn& in_gammaSum)
        {
            for (Eigen::Index iState = 0; iState < m_nStates; ++iState)
            {
                const ProbRow newCoefficients = m_coefficientSum[iState] / in_gammaSum[iState];
                const ProbMatrix newMeans = m_meanSum[iState].array().rowwise() / m_coefficientSum[iState].array();
                ProbTensor newCovariances = ProbTensor(m_nComponents);
                for (Eigen::Index iComponent = 0; iComponent < m_nComponents; ++iComponent)
                {
                    const ProbMatrix meansOuterProduct = newMeans.col(iComponent) * newMeans.col(iComponent).transpose();
                    newCovariances[iComponent] = m_covarianceSum[iState][iComponent] / m_coefficientSum[iState][iComponent] - meansOuterProduct;
                }
                m_gmms[iState] = GaussianMixtureModel<T>(newMeans, newCovariances, newCoefficients);
            }
            m_coefficientSum.assign(m_nStates, ProbRow::Zero(m_nComponents));
            m_meanSum.assign(m_nStates, ProbMatrix::Zero(m_nDimensions, m_nComponents));
            m_covarianceSum.assign(m_nStates, ProbTensor(m_nComponents, ProbMatrix::Zero(m_nDimensions, m_nDimensions)));
        }
        
        const ObsSeqType generate(const StateSeqType& in_hiddenSequence) const
        {
            const Eigen::Index seqLength = in_hiddenSequence.size();
            ProbMatrix observedSequence(m_nDimensions, seqLength);
            for (Eigen::Index iSeq = 0; iSeq < seqLength; ++iSeq)
            {
                observedSequence.col(iSeq) = m_gmms[in_hiddenSequence[iSeq]].generate();
            }
            return observedSequence;
        }
        
    private:
        const Eigen::Index m_nStates;
        const Eigen::Index m_nDimensions;
        const Eigen::Index m_nComponents;
        std::vector<GaussianMixtureModel<T>> m_gmms;
        std::vector<ProbRow> m_coefficientSum; //[nStates x [1 x nComponents]]
        std::vector<ProbMatrix> m_meanSum; //[nStates x [nDimensions x nComponents]]
        std::vector<ProbTensor> m_covarianceSum; //[nStates x [nComponents x [nDimensions x nDimensions]]]
    };
}

#endif /* ObservationsGMM_h */
