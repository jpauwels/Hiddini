//
//  GaussianMixtureModel.h
//  hiddini
//
//  Created by Johan Pauwels on 23/03/2017.
//
//

#ifndef GaussianMixtureModel_h
#define GaussianMixtureModel_h

#include "MultivariateGaussian.h"

namespace hiddini
{
    template<typename T>
    class GaussianMixtureModel
    {
    public:
        
        typedef typename MultivariateGaussian<T>::ProbRow ProbRow;
        typedef typename MultivariateGaussian<T>::ProbColumn ProbColumn;
        typedef typename MultivariateGaussian<T>::ProbMatrix ProbMatrix;
        typedef std::vector<ProbMatrix> ProbTensor;
        
        GaussianMixtureModel(const Eigen::Index in_nDimensions, const Eigen::Index in_nComponents)
        : m_nComponents(in_nComponents)
        , m_mixCoefficients(ProbRow::Constant(m_nComponents, 1./m_nComponents))
        , m_gaussians(m_nComponents, MultivariateGaussian<T>(in_nDimensions))
        {
        }
        
        // @param   in_means [nDimensions x nComponents]
        // @param   in_covariances [nComponents x [nDimensions x nDimensions]]
        // @param   in_mixCoefficients [1 x nComponents]
        GaussianMixtureModel(const ProbMatrix& in_means, const ProbTensor& in_covariances, const ProbRow& in_mixCoefficients = ProbRow())
        : m_nComponents(in_means.cols())
        , m_mixCoefficients(in_mixCoefficients)
        , m_gaussians(m_nComponents, MultivariateGaussian<T>(in_means.rows())), m_coeffBoundaries(m_nComponents)
        {
            if (in_mixCoefficients.size() == 0)
            {
                m_mixCoefficients = ProbRow::Constant(m_nComponents, 1./m_nComponents);
            }
            for (Eigen::Index iComponent = 0; iComponent < m_nComponents; ++iComponent)
            {
                m_gaussians[iComponent] = MultivariateGaussian<T>(in_means.col(iComponent), in_covariances[iComponent]);
            }
            
            // Cumulative sum of mixture probabilities used for sampling Gaussians according to mixture coefficient distribution
            std::partial_sum(m_mixCoefficients.data(), m_mixCoefficients.data()+m_nComponents, m_coeffBoundaries.data());
        }
        
        const Eigen::Index getNumComponents() const
        {
            return m_nComponents;
        }
        
        // @param   in_observationSequence [nDimensions x nTimeSteps]
        // @return  [1 x nTimeSteps]
        const ProbRow operator()(const ProbMatrix& in_observationSequence) const
        {
            return m_mixCoefficients * perComponentProbs(in_observationSequence);
        }
        
        // @param   in_observationSequence [nDimensions x nTimeSteps]
        // @return  [nComponents x nTimeSteps]
        const ProbMatrix posteriorComponentProbs(const ProbMatrix& in_observationSequence) const
        {
            ProbMatrix posteriorComponentProbs = perComponentProbs(in_observationSequence).array().colwise() * m_mixCoefficients.array().transpose();
            posteriorComponentProbs.array() *= 1 / posteriorComponentProbs.array().colwise().sum(); //CHECK Eigen L1 normalized builtin method?
            return posteriorComponentProbs;
        }
        
//        const ProbMatrix meanCentredObservations(const ProbMatrix& in_observationSequence, const Eigen::Index in_iComponent) const
//        {
//            return m_gaussians[in_iComponent].meanCentredObservations(in_observationSequence);
//        }
        
        const ProbColumn generate() const
        {
            //Sample discrete distribution m_mixCoefficients
            Eigen::Index mixComponent;
            (ProbRow::Random(1).cwiseAbs()[0] <= m_coeffBoundaries.array()).maxCoeff(&mixComponent);
            return m_gaussians[mixComponent].generate();
        }
        
        const ProbMatrix generate(const Eigen::Index in_nSamples) const
        {
            ProbRow probs = ProbRow::Random(in_nSamples).cwiseAbs();
            ProbMatrix samples(m_gaussians[0].m_nDimensions, in_nSamples);
            for (Eigen::Index iSample = 0; iSample < in_nSamples; ++iSample)
            {
                Eigen::Index mixComponent;
                (probs[iSample] <= m_coeffBoundaries.array()).maxCoeff(&mixComponent);
                samples.col(iSample) = m_gaussians[mixComponent].generate();
            }
            return samples;
        }
        
    protected:
        // @param   in_observationSequence [nDimensions x nTimeSteps]
        // @return  [nComponents x nTimeSteps]
        const ProbMatrix perComponentProbs(const ProbMatrix& in_observationSequence) const
        {
            ProbMatrix componentProbs(m_nComponents, in_observationSequence.cols());
            for (Eigen::Index iComponent = 0; iComponent < m_nComponents; ++iComponent)
            {
                componentProbs.row(iComponent) = m_gaussians[iComponent](in_observationSequence);
            }
            return componentProbs;
        }
        
    private:
        Eigen::Index m_nComponents;
        ProbRow m_mixCoefficients; //[1 x nComponents]
        std::vector<MultivariateGaussian<T>> m_gaussians;
        ProbRow m_coeffBoundaries;
    };
}

#endif /* GaussianMixtureModel_h */
