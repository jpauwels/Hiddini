//
//  ObservationsDiscrete.h
//  hiddini
//
//  Created by Johan Pauwels on 14/03/2017.
//
//

#ifndef ObservationsDiscrete_h
#define ObservationsDiscrete_h

#include <Eigen/Dense>

namespace hiddini
{
    template<typename T>
    class ObservationsDiscrete
    {
    public:
        typedef T ProbType;
        typedef Eigen::Matrix<T, 1, Eigen::Dynamic> ProbRow;
        typedef Eigen::Matrix<T, Eigen::Dynamic, 1> ProbColumn;
        typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> ProbMatrix;
        typedef Eigen::Matrix<Eigen::Index, 1, Eigen::Dynamic> StateSeqType;
        typedef Eigen::Matrix<Eigen::Index, 1, Eigen::Dynamic> ObsSeqType;
        
        ObservationsDiscrete(const Eigen::Index in_nStates, const Eigen::Index in_nSymbols)
        : m_nStates(in_nStates), m_nSymbols(in_nSymbols)
        , m_obsPmf(in_nStates, in_nSymbols), m_observationsSum(ProbMatrix::Zero(in_nStates, in_nSymbols))
        {
            m_obsPmf.setRandom();
            m_obsPmf = m_obsPmf.array().abs();
            m_obsPmf.array().colwise() *= 1 / m_obsPmf.rowwise().sum().array();
        }
        
        ObservationsDiscrete(const ProbMatrix& in_observationsPmf)
        : m_nStates(in_observationsPmf.rows()), m_nSymbols(in_observationsPmf.cols())
        , m_obsPmf(in_observationsPmf), m_observationsSum(ProbMatrix::Zero(m_nStates, m_nSymbols))
        {
        }
        
        const ProbMatrix operator()(const ObsSeqType& in_observationSequence) const
        {
            return m_obsPmf(Eigen::placeholders::all, in_observationSequence);
        }
        
    protected:
        friend class HMM<ObservationsDiscrete<T>>;
        
        Eigen::Index getNumStates() const
        {
            return m_nStates;
        }
        
        void reestimateEmissionParameters(const ObsSeqType& in_observationSequence, const ProbMatrix& in_gammas)
        {
            for (Eigen::Index iSymbol = 0; iSymbol < m_nSymbols; ++iSymbol)
            {
                m_observationsSum.col(iSymbol) += in_gammas(Eigen::placeholders::all, in_observationSequence.array() == iSymbol).rowwise().sum();
            }
        }
        
        void saveEmissionParameters(const ProbColumn& in_gammaSum)
        {
            m_obsPmf = m_observationsSum.array().colwise() / in_gammaSum.array();
            m_observationsSum.setZero();
        }
        
        const ObsSeqType generate(const StateSeqType& in_hiddenSequence) const
        {
            const Eigen::Index seqLength = in_hiddenSequence.size();
            ProbMatrix symbolBoundaries(m_nStates, m_nSymbols);
            symbolBoundaries.col(0) = m_obsPmf.col(0);
            for (Eigen::Index iCol = 1; iCol < m_nSymbols; ++iCol)
            {
                symbolBoundaries.col(iCol) = symbolBoundaries.col(iCol-1) + m_obsPmf.col(iCol);
            }
            ProbRow probs = ProbRow::Random(seqLength).cwiseAbs();
            ObsSeqType observedSequence(seqLength);
            for (Eigen::Index iSeq = 0; iSeq < seqLength; ++iSeq)
            {
                (probs[iSeq] <= symbolBoundaries.row(in_hiddenSequence[iSeq]).array()).maxCoeff(&observedSequence[iSeq]);
            }
            return observedSequence;
        }
        
    private:
        Eigen::Index m_nStates;
        Eigen::Index m_nSymbols;
        ProbMatrix m_obsPmf;
        ProbMatrix m_observationsSum;
    };
}

#endif /* ObservationsDiscrete_h */
