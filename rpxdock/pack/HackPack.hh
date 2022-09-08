#pragma once

#include <boost/foreach.hpp>
#include <random>

#include "rpxdock/pack/TwoBodyTable.hh"

namespace rpxdock {
namespace pack {

template <typename Float>
inline bool pass_metropolis(Float const& temperature, Float const& deltaE,
                            Float const& random_uniform) {
  if (deltaE < 0) {
    return true;
  } else {  // evaluate prob of substitution
    Float lnprob = deltaE / temperature;
    if (lnprob < 10.0) {
      Float probability = std::exp(-lnprob);
      if (probability > random_uniform) return true;
    }
  }
  return false;
}

struct HackPackOpts {
  int pack_n_iters = 1;
  float pack_iter_mult = 2.0;
  float hbond_weight = 2.0;
  float upweight_iface = 1.0;
  float upweight_multi_hbond = 0.0;
  float min_hb_quality_for_satisfaction = -0.6;
  bool use_extra_rotamers = true;
  int always_available_rotamers_level = 0;
  bool packing_use_rif_rotamers = true;
  bool add_native_scaffold_rots_when_packing = false;
  float rotamer_inclusion_threshold = -0.5;
  float rotamer_onebody_inclusion_threshold = 30.0;  // 5
  bool init_with_best_1be_rots = true;
  float user_rotamer_bonus_constant = -2;  //-2
  float user_rotamer_bonus_per_chi = -2;   // 2
  bool rescore_rots_before_insertion =
      true;  // this isn't a real flag, gets used in MyScoreBBActorVsRif
};
inline std::ostream& operator<<(std::ostream& out, HackPackOpts const& hpo) {
  out << "HackPackOpts:"
      << "\n  pack_iter_mult " << hpo.pack_iter_mult << "\n  hbond_weight "
      << hpo.hbond_weight << "\n  upweight_iface " << hpo.upweight_iface
      << "\n  upweight_multi_hbond " << hpo.upweight_multi_hbond
      << "\n  use_extra_rotamers " << hpo.use_extra_rotamers
      << "\n  always_available_rotamers_level "
      << hpo.always_available_rotamers_level << "\n  packing_use_rif_rotamers "
      << hpo.packing_use_rif_rotamers
      << "\n  add_native_scaffold_rots_when_packing "
      << hpo.add_native_scaffold_rots_when_packing
      << "\n  rotamer_inclusion_threshold " << hpo.rotamer_inclusion_threshold
      << "\n  rotamer_onebody_inclusion_threshold "
      << hpo.rotamer_onebody_inclusion_threshold
      << "\n  init_with_best_1be_rots " << hpo.init_with_best_1be_rots
      << "\n  user_rotamer_bonus_constant " << hpo.user_rotamer_bonus_constant
      << "\n  user_rotamer_bonus_per_chi" << hpo.user_rotamer_bonus_per_chi
      << "\n  rescore_rots_before_insertion "
      << hpo.rescore_rots_before_insertion

      << std::endl;
  return out;
}

struct HackPack {
  typedef std::pair<int32_t, float> RotInfo;
  typedef std::pair<int32_t, std::vector<RotInfo>> RotInfos;
  int nres_;                        // total res currently stored
  std::vector<RotInfos> res_rots_;  // iresapp + list of irottwob/onebody pairs
  std::vector<std::pair<int32_t, int32_t>>
      rot_list_;  // list of ireslocal / irotlocal pairs
  std::vector<int32_t> current_rots_, trial_best_rots_,
      global_best_rots_;  // current rotamer in local numbering
  std::mt19937 rng;
  shared_ptr<::rpxdock::pack::TwoBodyTable<float>> twob_;
  float score_, trial_best_score_, global_best_score_;
  HackPackOpts opts_;
  int32_t default_rot_num_;
  HackPack(
      // ::rpxdock::pack::TwoBodyTable<float> const & twob,
      HackPackOpts const& opts, int32_t default_rot_num,
      int seed_offset = 0  // mainly for threads
      )
      : nres_(0),
        rng(time(0) + seed_offset)
        // , twob_( twob )
        ,
        opts_(opts),
        default_rot_num_(default_rot_num) {}

  void reinitialize(shared_ptr<::rpxdock::pack::TwoBodyTable<float>> twob) {
    // Brian

    twob_ = twob;
    ALWAYS_ASSERT(twob_->nrot_ > 0);
    ALWAYS_ASSERT(twob_->nres_ > 0);
    ALWAYS_ASSERT(twob_->nrot_ < 99999);
    ALWAYS_ASSERT(twob_->nres_ < 99999);

    ////////////////////

    // todo: always add native rotamer and ALA/GLY as appropriate
    // should hopefully not deallocate memory
    rot_list_.clear();
    for (RotInfos& rotinfos : res_rots_) {
      rotinfos.first = -1;
      rotinfos.second.clear();
    }
    nres_ = 0;
  }
  template <class Int>
  bool using_rotamer(Int const& ires, Int const& irotglobal) {
    ALWAYS_ASSERT(0 <= ires && ires < twob_->all2sel_.shape()[0]);
    ALWAYS_ASSERT(0 <= irotglobal && irotglobal < twob_->all2sel_.shape()[1]);
    return twob_->all2sel_[ires][irotglobal] >= 0;
  }
  template <class Int>
  void add_tmp_rot(int const& ires, Int const& irotglobal,
                   float const& onebody_e) {
    // #pragma omp critical
    // {
    // 	std::cout << "================= add_tmp_rot " << ires << " " <<
    // irotglobal << " " << onebody_e << std::endl; 	print_rot_info();
    // }
    if (onebody_e > 10.0) {
      return;
    }
    ALWAYS_ASSERT(0 <= ires && ires < twob_->all2sel_.shape()[0]);
    ALWAYS_ASSERT(0 <= irotglobal && irotglobal < twob_->all2sel_.shape()[1]);
    int32_t irotlocal = twob_->all2sel_[ires][irotglobal];
    // std::cout << "irotlocal" << irotlocal << std::endl;
    if (irotlocal >= 0) {
      if (nres_ == 0 || res_rots_.at(nres_ - 1).first != ires) {
        ++nres_;
        if (res_rots_.size() < nres_) res_rots_.resize(nres_);
        res_rots_.at(nres_ - 1).first = ires;
        // always allow ALA as an option:
        int alarot = twob_->all2sel_[ires][default_rot_num_];
        if (alarot >= 0) {
          rot_list_.push_back(
              std::make_pair(nres_ - 1, res_rots_.at(nres_ - 1).second.size()));
          res_rots_.at(nres_ - 1).second.push_back(RotInfo(alarot, 0.0));
        }
      }
      rot_list_.push_back(
          std::make_pair(nres_ - 1, res_rots_.at(nres_ - 1).second.size()));
      res_rots_.at(nres_ - 1).second.push_back(RotInfo(irotlocal, onebody_e));
    } else {
      // std::cout << "Error!!!: Rotamer not in twobody energies " << irotglobal
      // << " " << ires << std::endl; static bool missingrotwarn = true; if(
      // missingrotwarn ){ 	#ifdef USE_OPENMP 	#pragma omp critical
      // #endif
      // 	{
      // 		std::cout << "WARNING: requested rotamer not in
      // TwoBodyTable, probably has bad 1BE, subsequent warnings will be
      // skipped: "
      // 		          << " irotglobal: " << irotglobal << std::endl;
      // 		missingrotwarn = false;
      // 		// std::exit(-1);
      // 	}
      // }
    }
  }

  float compute_energy_full(std::vector<int32_t> const& rots) const {
    // using namespace ObjexxFCL::format;
    // for( int i = 1; i < nres_; ++i ){
    // 		int   const iresglobal = res_rots_[i].first;
    // 		int   const irottwob   = res_rots_[i].second[ rots[i] ].first;
    // 		float const ionebody   = res_rots_[i].second[ rots[i] ].second;
    // 		int irotglobal = twob_->sel2all_[ iresglobal ][ irottwob ];
    // 		std::cout << "ONEBODY " << iresglobal << " " <<
    // rot_index_.resname(irotglobal) << irotglobal << " " << ionebody <<
    // std::endl;
    // 	}
    float score = 0.0;
    for (int ires = 0; ires < nres_; ++ires) {
      assert(0 <= ires && ires < rots.size());
      int32_t const irotlocal = rots.at(ires);
      assert(0 <= ires && ires < res_rots_.size());
      // std::cout << ires << " " << irotlocal << " " <<
      // res_rots_[ires].second.size() << std::endl;
      assert(0 <= irotlocal && irotlocal < res_rots_.at(ires).second.size());
      int32_t const iresglobal = res_rots_.at(ires).first;
      int32_t const irottwob = res_rots_.at(ires).second.at(irotlocal).first;
      float const ionebody = res_rots_.at(ires).second.at(irotlocal).second;
      score += ionebody;
      for (int jres = 0; jres < ires; ++jres) {
        assert(0 <= jres && jres < rots.size());
        int32_t const jrotlocal = rots.at(jres);
        assert(0 <= jres && jres < res_rots_.size());
        assert(0 <= jrotlocal && jrotlocal < res_rots_.at(jres).second.size());
        int32_t const jresglobal = res_rots_.at(jres).first;
        int32_t const jrottwob = res_rots_.at(jres).second.at(jrotlocal).first;
        float const jonebody = res_rots_.at(jres).second.at(jrotlocal).second;
        float const twobodye = twob_->twobody_rotlocalnumbering(
            iresglobal, jresglobal, irottwob, jrottwob);
        score += twobodye;
        // int irotglobal = twob_->sel2all_[ iresglobal ][ irottwob ];
        // int jrotglobal = twob_->sel2all_[ jresglobal ][ jrottwob ];
        // std::cout << "TWOBODY "
        //           << I(2,ires) << "/" << I(2,jres) << " "
        //           << I(3,iresglobal) << "/" << I(3,jresglobal) << " "
        //           << rot_index_.resname(irotglobal) << I(3,irottwob) << "/"
        //           << rot_index_.resname(jrotglobal) << I(3,jrottwob) << " "
        //           << F(7,3,twobodye) << std::endl;
      }
    }
    return score;
  }
  float compute_energy_delta(std::vector<int32_t> const& rots,
                             int32_t const& ilres,
                             int32_t const& ilrotnew) const {
    // using namespace ObjexxFCL::format;
    float delta = 0;
    int32_t const ilrotold = rots.at(ilres);
    int32_t const iresglobal = res_rots_.at(ilres).first;
    int32_t const irottwobold =
        res_rots_.at(ilres).second.at(rots.at(ilres)).first;
    float const ionebodyold =
        res_rots_.at(ilres).second.at(rots.at(ilres)).second;
    int32_t const irottwobnew = res_rots_.at(ilres).second.at(ilrotnew).first;
    float const ionebodynew = res_rots_.at(ilres).second.at(ilrotnew).second;
    delta -= ionebodyold;
    delta += ionebodynew;
    for (int j = 0; j < nres_; ++j) {
      if (j == ilres) continue;
      int32_t const jresglobal = res_rots_.at(j).first;
      int32_t const jrottwob = res_rots_.at(j).second.at(rots.at(j)).first;
      float const jonebody = res_rots_.at(j).second.at(rots.at(j)).second;
      float const twobodyeold = twob_->twobody_rotlocalnumbering(
          iresglobal, jresglobal, irottwobold, jrottwob);
      float const twobodyenew = twob_->twobody_rotlocalnumbering(
          iresglobal, jresglobal, irottwobnew, jrottwob);
      delta -= twobodyeold;
      delta += twobodyenew;
      // std::cout << "DELTA TWOB"
      //           << " ires "    << I(2,ilres   ) << "/" << I(3,iresglobal )
      //           << " jres "    << I(2,j       ) << "/" << I(3,jresglobal)
      //           << " irotold " << I(2,ilrotold) << "/" << I(3,irottwobold)
      //           << " irotnew " << I(2,ilrotnew) << "/" << I(3,irottwobnew)
      //           << " jrot "    << I(2,rots[j] ) << "/" << I(3,jrottwob)
      //           << " e " << F(7,3,twobodyeold)  << " " << F(7,3,twobodyenew)
      //           << std::endl;
    }
    if (-123460.0 > delta ||
        delta > 123460.0) {  // 10x energy cap per-rottable entry
      bool throwerr = false;
#ifdef USE_OPENMP
#pragma omp critical
#endif
      {
        std::cout << "crazy energy delta, indicates a problem.... res (scene "
                     "numbering) = "
                  << res_rots_.at(ilres).first << " " << delta << std::endl;
        static int errcount = 0;
        if (++errcount > 10) throwerr = true;
      }
      if (throwerr) throw std::logic_error("too many crazy energy deltas");
      return 9e9;
    }
    return delta;
  }
  int32_t randres() {
    std::uniform_int_distribution<> rand_idx(0, nres_ - 1);
    return rand_idx(rng);
  }
  int32_t randrot(int32_t const& ires) {
    assert(res_rots_.at(ires).second.size() > 0);
    // if( res_rots_[ires].second.size() == 1 ) return 0;
    std::uniform_int_distribution<> rand_idx(
        0, res_rots_.at(ires).second.size() - 1);
    return rand_idx(rng);
  }
  void randrot_not_current_uniform_res(int32_t& ires, int32_t& irot) {
    for (int k = 0; k < 1000; ++k) {
      ires = randres();
      if (res_rots_.at(ires).second.size() > 1) break;
    }
    ALWAYS_ASSERT(0 <= ires && ires < nres_);
    ALWAYS_ASSERT(res_rots_.at(ires).second.size() > 1);

    for (int k = 0; k < 1000; ++k) {
      irot = randrot(ires);
      if (irot != current_rots_.at(ires)) return;
    }
    ALWAYS_ASSERT(0 <= irot && irot < res_rots_.at(ires).second.size());
  }
  void randrot_not_current_uniform_rot(int32_t& ires, int32_t& irot) {
    std::uniform_int_distribution<> rand_idx(0, rot_list_.size() - 1);
    for (int i = 0; i < 1000; ++i) {
      int const irand = rand_idx(rng);
      ires = rot_list_.at(irand).first;
      irot = rot_list_.at(irand).second;
      if (res_rots_.at(ires).second.size() > 1 &&
          irot != current_rots_.at(ires))
        return;
    }
    std::cerr << "randrot_not_current_uniform_rot FAIL" << std::endl;
    std::exit(-1);
  }
  void random_substitution_test(float temperature) {
    std::uniform_real_distribution<float> runif(0, 1);

    int32_t ires, irot;
    randrot_not_current_uniform_rot(ires, irot);

    float delta = compute_energy_delta(current_rots_, ires, irot);
    // {
    // 	// std::cout << "SUB: " << ires << " " << irot << " " <<
    // res_rots_[ires].first << std::endl;
    // 	// std::cout << "==================================== old
    // ==========================================" << std::endl;
    // 	// for( int k = 0; k < current_rots_.size(); ++k ) std::cout << "currot
    // " << k << " " << current_rots_[k] << std::endl; 	float curfull =
    // compute_energy_full( current_rots_ ); 	int32_t tmp =
    // current_rots_[ires]; 	current_rots_[ires] = irot;
    // 	// std::cout << "==================================== new
    // ==========================================" << std::endl;
    // 	// for( int k = 0; k < current_rots_.size(); ++k ) std::cout << "currot
    // " << k << " " << current_rots_[k] << std::endl; 	float newfull =
    // compute_energy_full( current_rots_ ); 	current_rots_[ires] = tmp;
    // float test_delta = newfull-curfull;
    // 	// std::cout << delta << " " << test_delta << " " << curfull << " " <<
    // newfull << std::endl; 	if( fabs(delta-test_delta) > 0.01 ){
    // #pragma omp critical 		std::cout << "fabs(delta-test_delta) "
    // << fabs(delta-test_delta)
    // << " " << delta << " " << test_delta << std::endl;
    // 	}
    // }

    if (pass_metropolis(temperature, delta, runif(rng))) {
      current_rots_.at(ires) = irot;
      score_ += delta;
      if (score_ < trial_best_score_) {
        trial_best_score_ = score_;
        trial_best_rots_ = current_rots_;
      }
    }
  }
  void recover_trial_best() {
    score_ = trial_best_score_;
    current_rots_ = trial_best_rots_;
  }
  void assign_random_rots() {
    current_rots_.resize(nres_);
    for (int ires = 0; ires < nres_; ++ires) {
      current_rots_.at(ires) = randrot(ires);
      // std::cout << "starting rot " << ires << " " << current_rots_[ires] <<
      // std::endl;
      assert(0 <= current_rots_.at(ires) &&
             current_rots_.at(ires) < res_rots_.at(ires).second.size());
    }
  }
  void assign_best_obe_rots() {
    current_rots_.resize(nres_);
    for (int ilres = 0; ilres < nres_; ++ilres) {
      float best = 9e9;
      for (int ilrot = 0; ilrot < res_rots_.at(ilres).second.size(); ++ilrot) {
        float obe = res_rots_.at(ilres).second.at(ilrot).second;
        if (obe < best) {
          best = obe;
          current_rots_.at(ilres) = ilrot;
        }
      }
    }
  }
  void assign_initial_rots() {
    if (opts_.init_with_best_1be_rots) {
      assign_best_obe_rots();
    } else {
      assign_random_rots();
    }
  }
  void fill_result_rots(std::vector<std::pair<int32_t, int32_t>>& result_rots) {
    result_rots.clear();
    for (int i = 0; i < nres_; ++i) {
      int32_t iresglobal = res_rots_.at(i).first;
      int32_t irottwob =
          res_rots_.at(i).second.at(global_best_rots_.at(i)).first;
      ALWAYS_ASSERT(0 <= iresglobal && iresglobal < twob_->sel2all_.shape()[0]);
      if (irottwob < 0) {
#ifdef USE_OPENMP
#pragma omp critical
#endif
        {
          std::cout << "local rot num: " << i << " iresglobal: " << iresglobal
                    << " irottwob: " << irottwob << std::endl;
          print_rot_info();
          std::cerr << "debug...." << std::endl;
          std::exit(-1);
        }
      }
      ALWAYS_ASSERT(0 <= irottwob && irottwob < twob_->sel2all_.shape()[1]);
      int32_t irotglobal = twob_->sel2all_[iresglobal][irottwob];
      result_rots.push_back(std::make_pair(iresglobal, irotglobal));
    }
  }
  float pack(std::vector<std::pair<int32_t, int32_t>>& result_rots) {
    assert(res_rots_.size() >= nres_);
    for (int i = 0; i < nres_; ++i) {
      // std::cout << i << " " << res_rots_[i].second.size() << std::endl;
      assert(res_rots_.at(i).second.size() > 0);
    }

    assign_initial_rots();

    uint64_t nchoices = 1;
    for (int ires = 0; ires < nres_; ++ires) {
      ALWAYS_ASSERT_MSG(res_rots_.at(ires).second.size() > 0,
                        "no rotamers at designable position!");
      nchoices *= res_rots_.at(ires).second.size();
      if (nchoices > 1000000000000000ull) break;
    }
    if (nchoices == 1) {
      global_best_rots_ = current_rots_;
      fill_result_rots(result_rots);
      score_ = compute_energy_full(current_rots_);
      return score_;
    }

    int const ntrials = opts_.pack_n_iters;
    int const pack_iters = opts_.pack_iter_mult * rot_list_.size() + 10;
    global_best_score_ = 9e9;
    for (int k = 0; k < ntrials; ++k) {
      if (k > 0) assign_initial_rots();
      score_ = compute_energy_full(current_rots_);
      trial_best_score_ = score_;
      trial_best_rots_ = current_rots_;
      for (int i = 0; i < pack_iters; ++i) random_substitution_test(100.0);
      recover_trial_best();
      for (int i = 0; i < pack_iters; ++i) random_substitution_test(33.0);
      recover_trial_best();
      for (int i = 0; i < pack_iters; ++i) random_substitution_test(10.0);
      recover_trial_best();
      for (int i = 0; i < pack_iters; ++i) random_substitution_test(3.3);
      recover_trial_best();
      for (int i = 0; i < pack_iters; ++i) random_substitution_test(1.0);
      recover_trial_best();
      for (int i = 0; i < pack_iters; ++i) random_substitution_test(0.33);
      recover_trial_best();
      for (int i = 0; i < pack_iters; ++i) random_substitution_test(0.1);
      recover_trial_best();
      for (int i = 0; i < pack_iters; ++i) random_substitution_test(0.0);
      recover_trial_best();
      if (score_ < global_best_score_) {
        global_best_score_ = score_;
        global_best_rots_ = current_rots_;
      }
    }

    fill_result_rots(result_rots);

    return score_;
  }

  void print_rot_info() const {
    // using namespace ObjexxFCL::format;

    std::cout << "====== res_rots_ =======" << std::endl;
    for (int ilres = 0; ilres < this->nres_; ++ilres) {
      int ires = this->res_rots_[ilres].first;
      // std::cout << I(3,ilres) << " res " << I(3,ires);
      std::cout << ilres << " res " << ires;
      for (typename HackPack::RotInfo const& rinfo :
           this->res_rots_[ilres].second) {
        std::cout << " " << rinfo.first << "/" << rinfo.second;
      }
      std::cout << std::endl;
    }
    std::cout << "====== rot_list_ =======" << std::endl;
    for (int k = 0; k < this->rot_list_.size(); ++k) {
      int32_t ilres = this->rot_list_[k].first;
      int32_t ilrot = this->rot_list_[k].second;
      // std::cout << I(3,k) << " res: " << I(2,ilres) << "/" <<
      // I(3,this->res_rots_[ilres].first)
      //           << " rot: " << I(2,ilrot) << "/" <<
      //           I(3,this->res_rots_[ilres].second[ilrot].first)
      //           << " score: " << this->res_rots_[ilres].second[ilrot].second
      //           << std::endl;
      std::cout << k << " res: " << ilres << "/" << this->res_rots_[ilres].first
                << " rot: " << ilrot << "/"
                << this->res_rots_[ilres].second[ilrot].first
                << " score: " << this->res_rots_[ilres].second[ilrot].second
                << std::endl;
    }
  }
};

}  // namespace pack
}  // namespace rpxdock
