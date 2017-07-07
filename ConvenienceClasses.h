//
//  ConvenienceClasses.h
//  hiddini
//
//  Created by Johan Pauwels on 05/07/2017.
//
//

#ifndef ConvenienceClasses_h
#define ConvenienceClasses_h

#include "HMM.h"
#include "ObservationsRaw.h"
#include "ObservationsDiscrete.h"
#include "ObservationsGaussian.h"
#include "ObservationsGMM.h"
#include "ObservationsTemplateCosSim.h"

namespace hiddini
{
    template <typename T>
    class HMMRaw : public HMM<ObservationsRaw<T>>
    {
    public:
        typedef typename ObservationsRaw<T>::ProbType ProbType;
        typedef typename ObservationsRaw<T>::ProbRow ProbRow;
        typedef typename ObservationsRaw<T>::ProbColumn ProbColumn;
        typedef typename ObservationsRaw<T>::ProbMatrix ProbMatrix;
        typedef typename ObservationsRaw<T>::StateSeqType StateSeqType;
        typedef typename ObservationsRaw<T>::ObsSeqType ObsSeqType;
        
        HMMRaw(const ProbMatrix& in_transProbs, const ProbColumn& in_initProbs)
        : HMM<ObservationsRaw<T>>(ObservationsRaw<T>(in_transProbs.rows()), in_transProbs, in_initProbs) {}
    };

    template <typename T>
    class HMMDiscrete : public HMM<ObservationsDiscrete<T>>
    {
    public:
        typedef typename ObservationsDiscrete<T>::ProbType ProbType;
        typedef typename ObservationsDiscrete<T>::ProbRow ProbRow;
        typedef typename ObservationsDiscrete<T>::ProbColumn ProbColumn;
        typedef typename ObservationsDiscrete<T>::ProbMatrix ProbMatrix;
        typedef typename ObservationsDiscrete<T>::StateSeqType StateSeqType;
        typedef typename ObservationsDiscrete<T>::ObsSeqType ObsSeqType;
        
        HMMDiscrete(const Eigen::Index in_nStates, const Eigen::Index in_nSymbols)
        : HMM<ObservationsDiscrete<T>>(ObservationsDiscrete<T>(in_nStates, in_nSymbols)) {}
        
        HMMDiscrete(const ProbMatrix& in_obsProbs, const ProbMatrix& in_transProbs, const ProbColumn& in_initProbs)
        : HMM<ObservationsDiscrete<T>>(ObservationsDiscrete<T>(in_obsProbs), in_transProbs, in_initProbs) {}
    };

    template <typename T>
    class HMMGaussian : public HMM<ObservationsGaussian<T>>
    {
    public:
        typedef typename ObservationsGaussian<T>::ProbType ProbType;
        typedef typename ObservationsGaussian<T>::ProbRow ProbRow;
        typedef typename ObservationsGaussian<T>::ProbColumn ProbColumn;
        typedef typename ObservationsGaussian<T>::ProbMatrix ProbMatrix;
        typedef typename ObservationsGaussian<T>::ProbTensor ProbTensor;
        typedef typename ObservationsGaussian<T>::StateSeqType StateSeqType;
        typedef typename ObservationsGaussian<T>::ObsSeqType ObsSeqType;
        
        HMMGaussian(const Eigen::Index in_nStates, const Eigen::Index in_nDimensions)
        : HMM<ObservationsGaussian<T>>(ObservationsGaussian<T>(in_nStates, in_nDimensions)) {}
        
        HMMGaussian(const ProbMatrix& in_means, const ProbTensor& in_covariances, const ProbMatrix& in_transProbs, const ProbColumn& in_initProbs)
        : HMM<ObservationsGaussian<T>>(ObservationsGaussian<T>(in_means, in_covariances), in_transProbs, in_initProbs) {}
    };

    template <typename T>
    class HMMGMM : public HMM<ObservationsGMM<T>>
    {
    public:
        typedef typename ObservationsGMM<T>::ProbType ProbType;
        typedef typename ObservationsGMM<T>::ProbRow ProbRow;
        typedef typename ObservationsGMM<T>::ProbColumn ProbColumn;
        typedef typename ObservationsGMM<T>::ProbMatrix ProbMatrix;
        typedef typename ObservationsGMM<T>::ProbTensor ProbTensor;
        typedef typename ObservationsGMM<T>::StateSeqType StateSeqType;
        typedef typename ObservationsGMM<T>::ObsSeqType ObsSeqType;
        
        HMMGMM(const Eigen::Index in_nStates, const Eigen::Index in_nDimensions, const Eigen::Index in_nComponents)
        : HMM<ObservationsGMM<T>>(ObservationsGMM<T>(in_nStates, in_nDimensions, in_nComponents)) {}
        
        HMMGMM(std::vector<ProbMatrix> in_mixMeans, std::vector<ProbTensor> in_mixCovariances, std::vector<ProbRow> in_mixCoefficients, const ProbMatrix& in_transProbs, const ProbColumn& in_initProbs)
        : HMM<ObservationsGMM<T>>(ObservationsGMM<T>(in_mixMeans, in_mixCovariances, in_mixCoefficients), in_transProbs, in_initProbs) {}
    };

    template <typename T>
    class HMMTemplateCosSim : public HMM<ObservationsTemplateCosSim<T>>
    {
    public:
        typedef typename ObservationsTemplateCosSim<T>::ProbType ProbType;
        typedef typename ObservationsTemplateCosSim<T>::ProbRow ProbRow;
        typedef typename ObservationsTemplateCosSim<T>::ProbColumn ProbColumn;
        typedef typename ObservationsTemplateCosSim<T>::ProbMatrix ProbMatrix;
        typedef typename ObservationsTemplateCosSim<T>::StateSeqType StateSeqType;
        typedef typename ObservationsTemplateCosSim<T>::ObsSeqType ObsSeqType;
        
        HMMTemplateCosSim(const ProbMatrix& in_templates, const ProbMatrix& in_transProbs, const ProbColumn& in_initProbs)
        : HMM<ObservationsTemplateCosSim<T>>(ObservationsTemplateCosSim<T>(in_templates), in_transProbs, in_initProbs) {}
    };
}


#endif /* ConvenienceClasses_h */
