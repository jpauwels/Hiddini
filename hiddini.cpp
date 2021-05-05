/*<%
setup_pybind11(cfg)
cfg['compiler_args'] = ['-std=c++11']
cfg['parallel'] = True
cfg['include_dirs'] = ['eigen']
%>*/
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "HMM.h"
#include "ObservationsRaw.h"
#include "ObservationsDiscrete.h"
#include "ObservationsGaussian.h"
#include "ObservationsGMM.h"
#include "ObservationsTemplateCosSim.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


namespace py = pybind11;
using namespace pybind11::literals;
using namespace hiddini;

typedef double FloatT;

PYBIND11_MODULE(hiddini, m)
{    
    m.doc() = "Magical Markov models";
    
    typedef ObservationsRaw<FloatT> ObservationsRaw;
    typedef HMM<ObservationsRaw> HMMRaw;
    py::class_<HMMRaw>(m, "HMMRaw")
        .def(py::init([](const HMMRaw::ProbMatrix& in_transProbs, const HMMRaw::ProbColumn& in_initProbs) {return HMMRaw(ObservationsRaw(in_transProbs.rows()), in_transProbs, in_initProbs);}), "transition_probs"_a, "initialisation_probs"_a = HMMRaw::ProbColumn())
        .def("evaluate", &HMMRaw::evaluate)
        .def("decodeMAP", &HMMRaw::decodeMAP, "likelikhoods_sequence"_a)
        .def("decodeMAP_with_lattice", &HMMRaw::decodeMAPWithLattice, "likelikhoods_sequence"_a)
        .def("decodePMAP", &HMMRaw::decodePMAP, "likelikhoods_sequence"_a)
        .def("decodePMAP_with_lattice", &HMMRaw::decodePMAPWithLattice, "likelikhoods_sequence"_a)
        .def("decodePV", &HMMRaw::decodePV, "likelikhoods_sequence"_a)
        .def("decodePV_with_lattice", &HMMRaw::decodePVWithLattice, "likelikhoods_sequence"_a)
        .def("decodeMAP_with_medianOPC", &HMMRaw::decodeMAPWithMedianOPC, "likelikhoods_sequence"_a)
        .def("decode_with_PPD", &HMMRaw::decodeWithPPD, "likelikhoods_sequence"_a, "output_decoder"_a="MAP", "additional_decoder"_a="PMAP")
    ;

    typedef ObservationsDiscrete<FloatT> ObservationsDiscrete;
    py::class_<ObservationsDiscrete>(m, "ObservationsDiscrete")
        .def(py::init<const Eigen::Index, const Eigen::Index>(), "num_states"_a, "num_symbols"_a)
        .def(py::init<const ObservationsDiscrete::ProbMatrix&>(), "observations_pmf"_a)
        .def("__call__", &ObservationsDiscrete::operator(), "observed_sequence"_a)
    ;

    typedef HMM<ObservationsDiscrete> HMMDiscrete;
    py::class_<HMMDiscrete>(m, "HMMDiscrete")
        .def(py::init([](const Eigen::Index in_nStates, const Eigen::Index in_nSymbols) {return HMMDiscrete(ObservationsDiscrete(in_nStates, in_nSymbols));}), "num_states"_a, "num_symbols"_a)
        .def(py::init([](const HMMDiscrete::ProbMatrix& in_obsProbs, const HMMDiscrete::ProbMatrix& in_transProbs, const HMMDiscrete::ProbColumn& in_initProbs) {return HMMDiscrete(ObservationsDiscrete(in_obsProbs), in_transProbs, in_initProbs);}), "observation_probs"_a, "transition_probs"_a, "initialisation_probs"_a = HMMDiscrete::ProbColumn())
        .def("evaluate", &HMMDiscrete::evaluate, "observations_sequence"_a)
        .def("decodeMAP", &HMMDiscrete::decodeMAP, "observations_sequence"_a)
        .def("decodeMAP_with_lattice", &HMMDiscrete::decodeMAPWithLattice, "observations_sequence"_a)
        .def("decodePMAP", &HMMDiscrete::decodePMAP, "observations_sequence"_a)
        .def("decodePMAP_with_lattice", &HMMDiscrete::decodePMAPWithLattice, "observations_sequence"_a)
        .def("decodePV", &HMMDiscrete::decodePV, "observations_sequence"_a)
        .def("decodePV_with_lattice", &HMMDiscrete::decodePVWithLattice, "observations_sequence"_a)
        .def("decodeMAP_with_medianOPC", &HMMDiscrete::decodeMAPWithMedianOPC, "observations_sequence"_a)
        .def("decode_with_PPD", &HMMDiscrete::decodeWithPPD, "observations_sequence"_a, "output_decoder"_a="MAP", "additional_decoder"_a="PMAP")
        .def("train", (void (HMMDiscrete::*)(const HMMDiscrete::ObsSeqType&, const Eigen::Index, const FloatT, const bool)) &HMMDiscrete::train, "observations_sequence"_a, "max_iterations"_a, "tolerance"_a, "verbose"_a=true)
        .def("train", (void (HMMDiscrete::*)(const std::vector<HMMDiscrete::ObsSeqType>&, const Eigen::Index, const FloatT, const bool)) &HMMDiscrete::train, "observations_sequence_list"_a, "max_iterations"_a, "tolerance"_a, "verbose"_a=true)
        .def("generate", &HMMDiscrete::generate, "sequence_length"_a)
    ;
    
    typedef ObservationsGaussian<FloatT> ObservationsGaussian;
    py::class_<ObservationsGaussian>(m, "ObservationsGaussian")
        .def(py::init<const Eigen::Index, const Eigen::Index>(), "num_states"_a, "num_dimensions"_a)
        .def(py::init<const ObservationsGaussian::ProbMatrix&, const ObservationsGaussian::ProbTensor&>(), "means"_a, "covariances"_a)
        .def("__call__", &ObservationsGaussian::operator(), "observed_sequence"_a)
    ;

    typedef HMM<ObservationsGaussian> HMMGaussian;
    py::class_<HMMGaussian>(m, "HMMGaussian")
        .def(py::init([](const Eigen::Index in_nStates, const Eigen::Index in_nDimensions) {return HMMGaussian(ObservationsGaussian(in_nStates, in_nDimensions));}), "num_states"_a, "num_dimensions"_a)
        .def(py::init([](const ObservationsGaussian::ProbMatrix& in_means, const ObservationsGaussian::ProbTensor& in_covariances, const ObservationsGaussian::ProbMatrix& in_transProbs, const ObservationsGaussian::ProbColumn& in_initProbs) {return HMMGaussian(ObservationsGaussian(in_means, in_covariances), in_transProbs, in_initProbs);}), "means"_a, "covariances"_a, "transition_probs"_a, "initialisation_probs"_a = HMMGaussian::ProbColumn())
        .def("evaluate", &HMMGaussian::evaluate, "observations_sequence"_a)
        .def("decodeMAP", &HMMGaussian::decodeMAP, "observations_sequence"_a)
        .def("decodeMAP_with_lattice", &HMMGaussian::decodeMAPWithLattice, "observations_sequence"_a)
        .def("decodePMAP", &HMMGaussian::decodePMAP, "observations_sequence"_a)
        .def("decodePMAP_with_lattice", &HMMGaussian::decodePMAPWithLattice, "observations_sequence"_a)
        .def("decodePV", &HMMGaussian::decodePV, "observations_sequence"_a)
        .def("decodePV_with_lattice", &HMMGaussian::decodePVWithLattice, "observations_sequence"_a)
        .def("decodeMAP_with_medianOPC", &HMMGaussian::decodeMAPWithMedianOPC, "observations_sequence"_a)
        .def("decode_with_PPD", &HMMGaussian::decodeWithPPD, "observations_sequence"_a, "output_decoder"_a="MAP", "additional_decoder"_a="PMAP")
        .def("train", (void (HMMGaussian::*)(const HMMGaussian::ObsSeqType&, const Eigen::Index, const FloatT, const bool)) &HMMGaussian::train, "observations_sequence"_a, "max_iterations"_a, "tolerance"_a, "verbose"_a=true)
        .def("train", (void (HMMGaussian::*)(const std::vector<HMMGaussian::ObsSeqType>&, const Eigen::Index, const FloatT, const bool)) &HMMGaussian::train, "observations_sequence_list"_a, "max_iterations"_a, "tolerance"_a, "verbose"_a=true)
        .def("generate", &HMMGaussian::generate, "sequence_length"_a)
    ;
    
    typedef ObservationsGMM<FloatT> ObservationsGMM;
    py::class_<ObservationsGMM>(m, "ObservationsGMM")
        .def(py::init<const Eigen::Index, const Eigen::Index, const Eigen::Index>(), "num_states"_a, "num_dimensions"_a, "num_components"_a)
        .def(py::init<const std::vector<ObservationsGMM::ProbMatrix>&, const std::vector<ObservationsGMM::ProbTensor>&, const std::vector<ObservationsGMM::ProbRow>&>(), "mix_means"_a, "mix_covariances"_a, "mix_coefficients"_a = std::vector<ObservationsGMM::ProbRow>())
        .def("__call__", &ObservationsGMM::operator(), "observed_sequence"_a)
    ;

    typedef HMM<ObservationsGMM> HMMGMM;
    py::class_<HMMGMM>(m, "HMMGMM")
        .def(py::init([](const Eigen::Index in_nStates, const Eigen::Index in_nDimensions, const Eigen::Index in_nComponents) {return HMMGMM(ObservationsGMM(in_nStates, in_nDimensions, in_nComponents));}), "num_states"_a, "num_dimensions"_a, "num_components"_a)
        .def(py::init([](const std::vector<ObservationsGMM::ProbMatrix>& in_mixMeans, const std::vector<ObservationsGMM::ProbTensor>& in_mixCovariances, const std::vector<ObservationsGMM::ProbRow>& in_mixCoefficients, const ObservationsGMM::ProbMatrix& in_transProbs, const ObservationsGMM::ProbColumn& in_initProbs) {return HMMGMM(ObservationsGMM(in_mixMeans, in_mixCovariances, in_mixCoefficients), in_transProbs, in_initProbs);}), "mix_means"_a, "mix_covariances"_a, "mix_coefficients"_a, "transition_probs"_a, "initialisation_probs"_a = HMMGMM::ProbColumn())
        .def("evaluate", &HMMGMM::evaluate, "observations_sequence"_a)
        .def("decodeMAP", &HMMGMM::decodeMAP, "observations_sequence"_a)
        .def("decodeMAP_with_lattice", &HMMGMM::decodeMAPWithLattice, "observations_sequence"_a)
        .def("decodePMAP", &HMMGMM::decodePMAP, "observations_sequence"_a)
        .def("decodePMAP_with_lattice", &HMMGMM::decodePMAPWithLattice, "observations_sequence"_a)
        .def("decodePV", &HMMGMM::decodePV, "observations_sequence"_a)
        .def("decodePV_with_lattice", &HMMGMM::decodePVWithLattice, "observations_sequence"_a)
        .def("decodeMAP_with_medianOPC", &HMMGMM::decodeMAPWithMedianOPC, "observations_sequence"_a)
        .def("decode_with_PPD", &HMMGMM::decodeWithPPD, "observations_sequence"_a, "output_decoder"_a="MAP", "additional_decoder"_a="PMAP")
        .def("train", (void (HMMGMM::*)(const HMMGMM::ObsSeqType&, const Eigen::Index, const FloatT, const bool)) &HMMGMM::train, "observations_sequence"_a, "max_iterations"_a, "tolerance"_a, "verbose"_a=true)
        .def("train", (void (HMMGMM::*)(const std::vector<HMMGMM::ObsSeqType>&, const Eigen::Index, const FloatT, const bool)) &HMMGMM::train, "observations_sequence_list"_a, "max_iterations"_a, "tolerance"_a, "verbose"_a=true)
        .def("generate", &HMMGMM::generate, "sequence_length"_a)
    ;
    
    typedef ObservationsTemplateCosSim<FloatT> ObservationsTemplateCosSim;
    py::class_<ObservationsTemplateCosSim>(m, "ObservationsTemplateCosSim")
        .def(py::init<const ObservationsTemplateCosSim::ProbMatrix&>(), "templates"_a)
        .def("__call__", &ObservationsTemplateCosSim::operator(), "observed_sequence"_a)
    ;

    typedef HMM<ObservationsTemplateCosSim> HMMTemplateCosSim;
    py::class_<HMMTemplateCosSim>(m, "HMMTemplateCosSim")
        .def(py::init([](const HMMTemplateCosSim::ProbMatrix& in_templates, const HMMTemplateCosSim::ProbMatrix& in_transProbs, const HMMTemplateCosSim::ProbColumn& in_initProbs) {return HMMTemplateCosSim(ObservationsTemplateCosSim(in_templates), in_transProbs, in_initProbs);}), "templates"_a, "transition_probs"_a, "initialisation_probs"_a = HMMTemplateCosSim::ProbColumn())
        .def("evaluate", &HMMTemplateCosSim::evaluate)
        .def("decodeMAP", &HMMTemplateCosSim::decodeMAP, "observations_sequence"_a)
        .def("decodeMAP_with_lattice", &HMMTemplateCosSim::decodeMAPWithLattice, "observations_sequence"_a)
        .def("decodePMAP", &HMMTemplateCosSim::decodePMAP, "observations_sequence"_a)
        .def("decodePMAP_with_lattice", &HMMTemplateCosSim::decodePMAPWithLattice, "observations_sequence"_a)
        .def("decodePV", &HMMTemplateCosSim::decodePV, "observations_sequence"_a)
        .def("decodePV_with_lattice", &HMMTemplateCosSim::decodePVWithLattice, "observations_sequence"_a)
        .def("decodeMAP_with_medianOPC", &HMMTemplateCosSim::decodeMAPWithMedianOPC, "observations_sequence"_a)
        .def("decode_with_PPD", &HMMTemplateCosSim::decodeWithPPD, "observations_sequence"_a, "output_decoder"_a="MAP", "additional_decoder"_a="PMAP")
    ;

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
