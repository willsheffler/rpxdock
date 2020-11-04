import rpxdock as rp, numpy as np, itertools as it
import rpxdock.rosetta.triggers_init as rti

def make_ideal_hgh(helix):
   helix = helix.clone()
   nhres = len(helix.residues)
   rti.core.pose.remove_lower_terminus_type_from_pose_residue(helix, 1)
   rti.core.pose.remove_upper_terminus_type_from_pose_residue(helix, nhres)
   gly = rti.create_residue('GLY')
   hgh1, hgh2 = helix.clone(), helix.clone()
   hgh1.append_residue_by_bond(gly, build_ideal_geometry=True)
   hgh1.set_omega(nhres, 180.0)
   hgh2.prepend_polymer_residue_before_seqpos(gly, 1, build_ideal_geometry=True)
   hgh2.set_omega(1, 180.0)
   x1 = rp.motif.frames.stub_from_points(
      hgh1.residue(nhres + 1).xyz('N'),
      hgh1.residue(nhres + 1).xyz('CA'),
      hgh1.residue(nhres + 1).xyz('C')).reshape(4, 4)
   x2 = rp.motif.frames.stub_from_points(
      hgh2.residue(1).xyz('N'),
      hgh2.residue(1).xyz('CA'),
      hgh2.residue(1).xyz('C')).reshape(4, 4)

   xform = x1 @ np.linalg.inv(x2)
   xform = rti.rosetta_stub_from_numpy_stub(xform)
   xform = rti.numeric.xyzTransform_double_t(xform.M, xform.v)
   rti.core.scoring.motif.xform_pose(hgh2, xform)

   rti.core.scoring.motif.xform_pose(helix, xform)
   rti.core.pose.append_pose_to_pose(hgh1, helix, False)

   m = rti.protocols.idealize.IdealizeMover()
   # hgh1.dump_pdb('hgh_post.pdb')

   return hgh1

def create_ball_joint_db():
   helix = rti.get_pose_cached('tiny.pdb', rp.data.pdbdir)
   ngly = len(helix.residues) + 1
   hgh = make_ideal_hgh(helix)
   # hgh.dump_pdb('hgh.pdb')
   rama = rti.core.scoring.Ramachandran()
   sfxn = rti.core.scoring.get_score_function()

   # print('start default', sfxn(hgh))
   # sfxn.set_weight(st.fa_atr, 0.0)
   # sfxn.set_weight(st.fa_sol, 0.0)
   # sfxn.set_weight(st.ref, 0.0)
   # print(sfxn.weights())
   # print_energies(hgh, sfxn)

   nsamp = 1000000

   binner = rp.Xbin(0.25, 1.0)

   stot, srama, dofs = (list() for l in range(3))
   keys = list()
   for i in range(nsamp):
      dof = [np.random.uniform(360.0) for i in range(6)]
      dof[1] = np.random.normal(180.0, 5.0)
      dof[4] = np.random.normal(180.0, 5.0)
      hgh.set_psi(ngly - 1, dof[0])
      hgh.set_omega(ngly - 1, dof[1])
      hgh.set_phi(ngly, dof[2])
      hgh.set_psi(ngly, dof[3])
      hgh.set_omega(ngly, dof[4])
      hgh.set_phi(ngly + 1, dof[5])
      rama_score = (rama.eval_rama_score_residue(hgh.residue(ngly - 1)) +
                    rama.eval_rama_score_residue(hgh.residue(ngly + 0)) +
                    rama.eval_rama_score_residue(hgh.residue(ngly + 1)) +
                    rama.eval_rama_score_residue(hgh.residue(ngly + 1)) +
                    rama.eval_rama_score_residue(hgh.residue(ngly + 1)))
      if rama_score > -1.5: continue

      s = sfxn(hgh)
      stot.append(s)
      srama.append(rama_score)
      dofs.append(dofs)
      x1 = rp.motif.frames.stub_from_points(
         hgh.residue(ngly - 7).xyz('N'),
         hgh.residue(ngly - 7).xyz('CA'),
         hgh.residue(ngly - 7).xyz('C')).reshape(4, 4)
      x2 = rp.motif.frames.stub_from_points(
         hgh.residue(ngly + 7).xyz('N'),
         hgh.residue(ngly + 7).xyz('CA'),
         hgh.residue(ngly + 7).xyz('C')).reshape(4, 4)
      xform = np.linalg.inv(x1) @ x2
      key = binner.key_of(xform)
      keys.append(key[0])

   print('-------------------------')
   print('N', nsamp, len(dofs))
   print('frac rama', len(dofs) / nsamp)
   print('min', np.min(srama))
   print('mean', np.mean(srama))
   print('stddev', np.std(srama))
   print(len(dofs), len(keys))

   fname = '' + str(np.random.random())[:-8] + '.pickle'
   rp.dump([stot, srama, keys, dofs], fname)

def print_energies(pose, sfxn):
   st = rti.core.scoring.ScoreType
   print('total', sfxn(pose))
   e = pose.energies()
   print('fa_atr', e.total_energies()[st.fa_atr])
   print('fa_rep', e.total_energies()[st.fa_rep])
   print('fa_sol', e.total_energies()[st.fa_sol])
   print('fa_intra_rep', e.total_energies()[st.fa_intra_rep])
   print('fa_elec', e.total_energies()[st.fa_elec])
   print('pro_close', e.total_energies()[st.pro_close])
   print('hbond_sr_bb', e.total_energies()[st.hbond_sr_bb])
   print('hbond_lr_bb', e.total_energies()[st.hbond_lr_bb])
   print('hbond_bb_sc', e.total_energies()[st.hbond_bb_sc])
   print('hbond_sc', e.total_energies()[st.hbond_sc])
   print('dslf_fa13', e.total_energies()[st.dslf_fa13])
   print('rama', e.total_energies()[st.rama])
   print('omega', e.total_energies()[st.omega])
   print('fa_dun', e.total_energies()[st.fa_dun])
   print('p_aa_pp', e.total_energies()[st.p_aa_pp])
   print('yhh_planarity', e.total_energies()[st.yhh_planarity])
   print('ref', e.total_energies()[st.ref])
