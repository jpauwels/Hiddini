//
//  ObservationsTemplateCosSim.h
//  hiddini
//
//  Created by Johan Pauwels on 17/03/2017.
//
//

#ifndef ObservationsTemplateCosSim_h
#define ObservationsTemplateCosSim_h

#include <Eigen/Dense>

namespace hiddini
{
    template<typename T>
    class ObservationsTemplateCosSim
    {
    public:
        typedef T ProbType;
        typedef Eigen::Matrix<T, 1, Eigen::Dynamic> ProbRow;
        typedef Eigen::Matrix<T, Eigen::Dynamic, 1> ProbColumn;
        typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> ProbMatrix;
        typedef Eigen::Matrix<Eigen::Index, 1, Eigen::Dynamic> StateSeqType;
        typedef ProbMatrix ObsSeqType;
        
        // @param   in_templates [nStates x nDimensions]
        ObservationsTemplateCosSim(const ProbMatrix& in_templates)
        : m_nStates(in_templates.rows()), m_nDimensions(in_templates.cols())
        , m_templates(in_templates), m_templatesNorm(m_templates.rowwise().norm())
        {
        }
        
        // @param   in_observationSequence [nDimensions x nObservations]
        // @return  [nStates x nObservations]
        const ProbMatrix operator()(const ObsSeqType& in_observationSequence) const
        {
            const ProbRow observationNorm = in_observationSequence.colwise().norm(); //[1 x nObservations]
            ProbMatrix obsProbs = (m_templates * in_observationSequence).cwiseQuotient(m_templatesNorm * observationNorm); //[nStates x nObservations]
            for (Eigen::Index iNorm = 0; iNorm < observationNorm.size(); ++iNorm)
            {
                if (observationNorm[iNorm] == 0)
                {
                    obsProbs.col(iNorm).setOnes();
                }
            }
            //obsProbs(observationNorm.array() == 0, Eigen::placeholders::all) = ProbColumn::Zero(m_nDimensions); TODO report Eigen
            return obsProbs; //TODO handle templateNorm == 0
        }
        
    protected:
        friend class HMM<ObservationsTemplateCosSim<T>>;
        
        Eigen::Index getNumStates() const
        {
            return m_nStates;
        }
        
    private:
        const Eigen::Index m_nStates;
        const Eigen::Index m_nDimensions;
        const ProbMatrix m_templates; //[nStates x nDimensions]
        const ProbColumn m_templatesNorm; //[nStates x 1]
    };
}

#endif /* ObservationsTemplateCosSim_h */
