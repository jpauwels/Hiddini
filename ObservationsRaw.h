//
//  ObservationsRaw.h
//  hiddini
//
//  Created by Johan Pauwels on 03/03/2017.
//
//

#ifndef ObservationsRaw_h
#define ObservationsRaw_h

#include <Eigen/Dense>

namespace hiddini
{
    template<typename T>
    class ObservationsRaw
    {
    public:
        typedef T ProbType;
        typedef Eigen::Matrix<T, 1, Eigen::Dynamic> ProbRow;
        typedef Eigen::Matrix<T, Eigen::Dynamic, 1> ProbColumn;
        typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> ProbMatrix;
        typedef Eigen::Matrix<Eigen::Index, 1, Eigen::Dynamic> StateSeqType;
        typedef ProbMatrix ObsSeqType;
        
        ObservationsRaw(const Eigen::Index in_nStates)
        : m_nStates(in_nStates)
        {
        }
        
        const ProbMatrix operator()(const ObsSeqType& in_obsProbSequence) const
        {
            return in_obsProbSequence;
        }
        
    protected:
        friend class HMM<ObservationsRaw<T>>;
        
        Eigen::Index getNumStates() const
        {
            return m_nStates;
        }
                
    private:
        const Eigen::Index m_nStates;
    };
}

#endif /* ObservationsRaw_h */
