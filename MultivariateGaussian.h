//
//  MultivariateGaussian.h
//  hiddini
//
//  Created by Johan Pauwels on 27/03/2017.
//
//

#ifndef MultivariateGaussian_h
#define MultivariateGaussian_h

#include <Eigen/Dense>
#include <random>

namespace hiddini
{
    template<typename T>
    class MultivariateGaussian
    {
    public:
        typedef Eigen::Matrix<T, 1, Eigen::Dynamic> ProbRow;
        typedef Eigen::Matrix<T, Eigen::Dynamic, 1> ProbColumn;
        typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> ProbMatrix;
        
        MultivariateGaussian(const Eigen::Index in_nDimensions)
        : m_nDimensions(in_nDimensions)
        , m_mean(m_nDimensions)
        , m_invCovariance(m_nDimensions, m_nDimensions)
        {
            m_mean.setRandom();
            // Create positive definite covariance matrix
            ProbMatrix covariance = ProbMatrix::Random(m_nDimensions, m_nDimensions);
            covariance = (covariance + covariance.transpose()).eval() / T(2); // make symmetrical first
            covariance = (covariance * covariance.transpose()).eval(); // then make Cholesky decomposable
            m_constant = 1 / std::sqrt(covariance.determinant() * std::pow(2*M_PI, m_nDimensions));
            m_invCovariance = -T(1)/T(2) * covariance.inverse();
        }
        
        MultivariateGaussian(const ProbColumn& in_mean, const ProbMatrix& in_covariance)
        : m_nDimensions(in_mean.size()), m_mean(in_mean)
        , m_constant(1 / std::sqrt(in_covariance.determinant() * std::pow(2*M_PI, m_nDimensions)))
        , m_invCovariance(-T(1)/T(2) * in_covariance.inverse())
        {
            //Check covariance matrix for positive (semi-)definiteness
            Eigen::LLT<ProbMatrix> cholSolver(in_covariance);
            if (cholSolver.info() == Eigen::Success)
            {
                // Positive definite
                m_stdNormTransform = cholSolver.matrixL();
            }
            else
            {
                Eigen::SelfAdjointEigenSolver<ProbMatrix> eigenSolver(in_covariance);
                if ((eigenSolver.eigenvalues().array() >= 0).all())
                {
                    // Positive semi-definite
                    m_stdNormTransform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
                    // Is it necessary to check if smallest eigenvalue is tiny, but negative? Fix: add .cwiseMax(0) to eigenvalues()
                    // Why check eigenvalues()(0)/eigenvalues()(n-1) > machine_precision in https://forum.kde.org/viewtopic.php?f=74&t=120899
                }
                else
                {
                    throw std::runtime_error("The given covariance matrix is not positive semi-definite");
                }
            }
            
            // Standard normal random number generator
            std::random_device seeder;
            std::mt19937 generator(seeder());
            std::normal_distribution<T> stdNorm(0, 1);
            m_stdNormGen = std::bind(stdNorm, std::ref(generator));
        }
        
        // @param in_observationSequence [nDimensions x nTimeSteps]
        // @return [1 x nTimeSteps]
        const ProbRow operator()(const ProbMatrix& in_observationSequence) const
        {
            ProbMatrix zeroMeanObs = in_observationSequence.colwise() - m_mean; //[nDimensions x nTimeSteps]
            return m_constant * zeroMeanObs.cwiseProduct(m_invCovariance * zeroMeanObs).colwise().sum().array().exp();
        }
        
//        const ProbMatrix meanCentredObservations(const ProbMatrix& in_observationSequence) const
//        {
//            return in_observationSequence.array().colwise() - m_mean.array();
//        }
//        
        const ProbColumn generate() const
        {
            return m_mean + m_stdNormTransform * ProbColumn::NullaryExpr(m_nDimensions, m_stdNormGen);
        }
        
        const ProbMatrix generate(const Eigen::Index in_nSamples) const
        {
            return m_mean + (m_stdNormTransform * ProbMatrix::NullaryExpr(m_nDimensions, in_nSamples, m_stdNormGen)).colwise();
        }
        
    private:
        Eigen::Index m_nDimensions;
        ProbColumn m_mean; //[nDimensions x 1]
        T m_constant; //scalar
        ProbMatrix m_invCovariance; //[nDimensions x nDimensions]
        std::function<T()> m_stdNormGen;
        ProbMatrix m_stdNormTransform;
    };
}

#endif /* MultivariateGaussian_h */
