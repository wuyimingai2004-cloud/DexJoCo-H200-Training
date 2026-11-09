"""Joint/A2V recipe plus legacy IDM/FDM compatibility on the tiny real WAM."""
from pathlib import Path
import sys
import unittest
from collections import Counter
from dataclasses import replace
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _rsi_cpu_stubs
import _rsi_harness as H
import torch
from test_packed_ar import packed_config, train_req, window, run_packed, close, capture_calls, Model
from thinker_wm.models import temporal_training as TT
from thinker_wm.models.temporal_layout import ROLE_GOAL, ROLE_TARGET, token_timesteps
from thinker_wm.models.inference_api import WAMInference, GenerationSettings, Observation, GoalActionOutput

MIX = dict(joint=.6, idm=.2, fdm=.2)
TWO_MODE_MIX = dict(joint=.6, a2v=.4)

def config(**kw):
    return packed_config(enable_depth=True, depth_condition_mode="rgb_only", training_mode_probs=MIX, **kw)

class JointA2VRecipe(unittest.TestCase):
    def test_template_and_sampling_exclude_idm(self):
        from omegaconf import OmegaConf
        fragment = OmegaConf.load(Path(__file__).resolve().parents[1] / 'tests/fixtures/training_temporal.yaml')
        kwargs = TT.motus_temporal_kwargs(fragment)
        self.assertEqual(kwargs['training_mode_probs'], TWO_MODE_MIX)
        cfg = replace(config(), training_mode_probs=kwargs['training_mode_probs'])
        TT.validate_temporal_config(cfg)
        counts = Counter(TT.sample_step_decision(cfg,s,True,False).mode for s in range(10000))
        self.assertEqual(set(counts), {TT.MODE_JOINT, TT.MODE_WORLD})
        self.assertLess(abs(counts[TT.MODE_WORLD] / 10000 - .4), .02)
        legacy = replace(cfg, training_mode_probs=dict(joint=.6,idm=0.,fdm=.4))
        for s in (0,1,50,52000):
            self.assertEqual(TT.sample_step_decision(cfg,s,True,False), TT.sample_step_decision(legacy,s,True,False))
        for bad in (dict(joint=.6,a2v=.4,fdm=0.), dict(joint=.6,a2v=.3,idm=.1),
                    dict(joint=.6,a2v=.3), dict(joint=float('inf'),a2v=.4), dict(joint=1.1,a2v=-.1)):
            with self.assertRaises(ValueError):
                TT.validate_temporal_config(replace(cfg,training_mode_probs=bad))

    def test_sampled_two_modes_four_blocks_history_and_backward(self):
        cfg = replace(config(num_video_frames=16,action_t_downsample=1,control_hz=50.,history_enable=True),
                      training_mode_probs=TWO_MODE_MIX)
        m = H.Harness(cfg).train()
        b = H.batch(depth=True,frames=16,history_slots=3)
        ctx = m.prepare_training_context(**b)
        calls = capture_calls(m)
        decisions = {}
        for s in range(100):
            d = TT.sample_step_decision(cfg,s,True,False)
            decisions.setdefault(d.mode,(s,d))
        self.assertEqual(set(decisions), {TT.MODE_JOINT,TT.MODE_WORLD})
        for mode,(s,d) in decisions.items():
            m.zero_grad(set_to_none=True)
            out = m.training_step(**b,training_context=ctx,dropout_seed_step=s)
            self.assertEqual(out['temporal_idm_mode'],0)
            self.assertEqual(out['temporal_a2v_mode'],int(mode==TT.MODE_WORLD))
            self.assertGreater(float(out['video_loss']),0)
            self.assertGreater(float(out['depth_loss']),0)
            self.assertEqual(float(out['action_loss'])==0,mode==TT.MODE_WORLD)
            tag = 'a2v' if mode==TT.MODE_WORLD else 'joint'
            blocks = {k.split('/')[1] for k in out if k.startswith(f'block_{tag}/')}
            want = {f'video_b{b}' for b in range(1,5)} | {f'depth_b{b}' for b in range(5)}
            if mode==TT.MODE_JOINT:
                want |= {f'action_b{b}' for b in range(1,5)}
            self.assertEqual(blocks, want)
            v = [out[f'block_{tag}/video_b{b}'] for b in range(1,5)]
            self.assertAlmostEqual(sum(v)/4, float(out['video_loss']), places=4)
            if mode==TT.MODE_JOINT:
                a = [out[f'block_joint/action_b{b}'] for b in range(1,5)]
                self.assertAlmostEqual(sum(a)/4, float(out['action_loss']), places=4)
            # D0 is supervised exactly once: five equal-weight depth latents.
            dep = [out[f'block_{tag}/depth_b{b}'] for b in range(5)]
            self.assertAlmostEqual(sum(dep)/5, float(out['depth_loss']), places=4)
            plan = TT.pack_plan(d.request)
            self.assertEqual([s for s,_,r in plan.depth if r==ROLE_TARGET].count(0), 1)
            if mode==TT.MODE_WORLD:
                # A2V commands are clean inputs, so their per-token timestep is 0.
                self.assertFalse(any(r==ROLE_TARGET for _,_,r in plan.action))
            self.assertFalse(close(calls[-1]['depth'][:,:,0],ctx['clean_depth_latent'][:,:,0]))
            if mode==TT.MODE_WORLD:
                torch.testing.assert_close(calls[-1]['action'],ctx['actions'])
            else:
                # The clean prefix is recorded data, not a sampled earlier block.
                for i,(src,_,role) in enumerate(plan.action):
                    if role != ROLE_TARGET:
                        torch.testing.assert_close(calls[-1]['action'][:,i],ctx['actions'][:,src])
            out['total_loss'].backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0
                                for p in m.world_model.action_branch.parameters()))
            g = m.world_model.history_encoder.logit_bias.grad
            self.assertIsNotNone(g)
            self.assertTrue(torch.isfinite(g).all())

    def test_a2v_actions_affect_future_rgb_depth_but_not_d0(self):
        req = train_req(action_role='world',depth_first_frame_conditioned=False)
        clean,noisy = window(8),window(9)
        state = torch.randn(2,8)
        base = run_packed(req,clean,noisy,state)
        modified = {k:v.clone() for k,v in clean.items()}
        modified['action'][:,:16] += 10
        changed = run_packed(req,modified,noisy,state)
        self.assertTrue(close(base[2][:,:,0],changed[2][:,:,0]))
        self.assertFalse(close(base[0][:,:,1],changed[0][:,:,1]))
        self.assertFalse(close(base[2][:,:,1],changed[2][:,:,1]))
        modified = {k:v.clone() for k,v in clean.items()}
        modified['action'][:,16:] += 10
        later = run_packed(req,modified,noisy,state)
        self.assertTrue(close(base[0][:,:,:2],later[0][:,:,:2]))
        self.assertTrue(close(base[2][:,:,:2],later[2][:,:,:2]))

class ModeConfig(unittest.TestCase):
    def test_validation_and_yaml(self):
        from omegaconf import OmegaConf
        cfg = config()
        TT.validate_temporal_config(cfg)
        self.assertEqual(TT.motus_temporal_kwargs(OmegaConf.create({'model': {'wm': {'training_mode_probs': MIX}}}))['training_mode_probs'], MIX)
        for bad in (dict(joint=.8, idm=.2), dict(joint=.7, idm=.2, fdm=.2), dict(joint=float('nan'), idm=.2, fdm=.2), dict(joint=1.2, idm=-.2, fdm=0)):
            with self.assertRaises(ValueError):
                TT.validate_temporal_config(replace(cfg, training_mode_probs=bad))
        for bad in (replace(cfg, world_mode_prob=.1), replace(cfg, temporal_mode='bidirectional')):
            with self.assertRaises(ValueError):
                TT.validate_temporal_config(bad)

    def test_sampling_resume_and_eval(self):
        cfg = config()
        counts = Counter(TT.sample_step_decision(cfg, s, True, False).mode for s in range(10000))
        for k,p in MIX.items():
            self.assertLess(abs(counts[TT.MODE_NAMES[k]] / 10000 - p), .02)
        for s in (0,1,50,52000):
            self.assertEqual(TT.sample_step_decision(cfg,s,True,False), TT.sample_step_decision(cfg,s,True,False))
            self.assertEqual(TT.sample_step_decision(cfg,s,False,False).mode, TT.MODE_JOINT)
        old = replace(cfg, training_mode_probs=None, world_mode_prob=1.)
        self.assertEqual(TT.sample_step_decision(old,0,True,False).mode, TT.MODE_WORLD)

    def test_mode_request_agreement(self):
        r = train_req(video_role='goal',depth_first_frame_conditioned=False)
        with self.assertRaises(ValueError):
            TT.StepDecision(r, TT.MODE_JOINT)
        with self.assertRaises(ValueError):
            replace(r, action_role='world')
        self.assertEqual(TT.pack_plan(r).video, ((0,0,1),(1,1,ROLE_GOAL),(2,2,ROLE_GOAL)))

class ModeTraining(unittest.TestCase):
    def test_all_modes_four_blocks_backward(self):
        for name,mode in TT.MODE_NAMES.items():
            with self.subTest(mode=name):
                cfg = config(num_video_frames=16, action_t_downsample=1, control_hz=50.,
                             modality_dropout_video=1., modality_dropout_action=1.)
                m=H.Harness(cfg).train()
                r=TT.sample_step_decision(cfg,0,False,False).request
                r=replace(r,action_role='world' if name=='fdm' else 'policy',video_role='goal' if name=='idm' else 'predict')
                b=H.batch(depth=True,frames=16)
                ctx=m.prepare_training_context(**b)
                calls=capture_calls(m)
                out=m.training_step(**b,training_context=ctx,temporal_decision=TT.StepDecision(r,mode),dropout_seed_step=0)
                self.assertEqual(out['modality_dropped_video'],0)
                self.assertEqual(out['modality_dropped_action'],0)
                self.assertTrue(torch.isfinite(out['total_loss']))
                self.assertGreater(float(out['depth_loss']),0)
                if name=='idm':
                    self.assertEqual(float(out['video_loss']),0)
                    self.assertGreater(float(out['action_loss']),0)
                    torch.testing.assert_close(calls[-1]['video'],ctx['clean_full_latent'])
                    self.assertFalse(close(calls[-1]['depth'][:,:,0],ctx['clean_depth_latent'][:,:,0]))
                elif name=='fdm':
                    self.assertEqual(float(out['action_loss']),0)
                    self.assertGreater(float(out['video_loss']),0)
                else:
                    self.assertGreater(float(out['action_loss']),0)
                    self.assertGreater(float(out['video_loss']),0)
                out['total_loss'].backward()
                grads=[p.grad for p in m.world_model.parameters() if p.grad is not None]
                self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))

    def test_cannot_override_rgb_only_with_gt_depth(self):
        m=H.Harness(config()).train()
        r=TT.sample_step_decision(m.config,0,False,True).request
        with self.assertRaisesRegex(ValueError,'depth conditioning'):
            m.training_step(**H.batch(depth=True),temporal_decision=TT.StepDecision(r,TT.MODE_JOINT))

    def test_mode_switches_with_history_optimizer_steps(self):
        cfg=config(history_enable=True, action_t_downsample=1, control_hz=50.)
        m=H.Harness(cfg).train()
        opt=torch.optim.AdamW(m.world_model.parameters(),lr=1e-5)
        b=H.batch(depth=True,history_slots=3)
        base=TT.sample_step_decision(cfg,0,False,False).request
        for name in ('joint','idm','fdm','joint','idm','fdm'):
            r=replace(base,action_role='world' if name=='fdm' else 'policy',
                      video_role='goal' if name=='idm' else 'predict')
            opt.zero_grad(set_to_none=True)
            out=m.training_step(**b,temporal_decision=TT.StepDecision(r,TT.MODE_NAMES[name]),dropout_seed_step=0)
            out['total_loss'].backward()
            grad=m.world_model.history_encoder.logit_bias.grad
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())
            opt.step()
            self.assertTrue(all(torch.isfinite(p).all() for p in m.world_model.parameters()))

    def test_goal_timestep_zero_depth_stays_noisy(self):
        r=train_req(video_role='goal',depth_first_frame_conditioned=False)
        # Segment constructors also govern the real world-model token timesteps.
        from thinker_wm.models.temporal_layout import Segment
        plan=TT.pack_plan(r)
        def seg(entries):
            return Segment(torch.tensor([e[1] for e in entries]),torch.tensor([e[2] for e in entries]),torch.zeros(len(entries),3))
        t=torch.tensor([[600.]])
        self.assertTrue((token_timesteps(t,seg(plan.video))==0).all())
        self.assertTrue((token_timesteps(t,seg(plan.depth))==600).all())

class ModeCausality(unittest.TestCase):
    def test_idm_d0_and_future_goal_isolation(self):
        req=train_req(video_role='goal',depth_first_frame_conditioned=False)
        clean,noisy=window(8),window(9)
        state=torch.randn(2,8)
        base=run_packed(req,clean,noisy,state)
        modified={k:v.clone() for k,v in clean.items()}
        modified['video'][:,:,1:]+=10
        changed=run_packed(req,modified,noisy,state)
        self.assertTrue(close(base[2][:,:,0],changed[2][:,:,0]),'D0 must ignore all future goals')
        self.assertFalse(close(base[1][:,:16],changed[1][:,:16]),'IDM action must read its own visual goal')
        modified={k:v.clone() for k,v in clean.items()}
        modified['video'][:,:,2]+=10
        later=run_packed(req,modified,noisy,state)
        self.assertTrue(close(base[1][:,:16],later[1][:,:16]),'future goal must not change earlier action')
        self.assertTrue(close(base[2][:,:,:2],later[2][:,:,:2]))

    def test_idm_no_clean_action_answer_leak(self):
        req=train_req(video_role='goal',depth_first_frame_conditioned=False)
        clean,noisy=window(8),window(9)
        state=torch.randn(2,8)
        base=run_packed(req,clean,noisy,state)
        modified={k:v.clone() for k,v in clean.items()}
        modified['action'][:,:16]+=100
        changed=run_packed(req,modified,noisy,state)
        self.assertTrue(close(base[1][:,:16],changed[1][:,:16]))
        self.assertTrue(close(base[2][:,:,:2],changed[2][:,:,:2]))
        self.assertFalse(close(base[1][:,16:],changed[1][:,16:]),'later block should use previous actions')

    def test_idm_packed_matches_separate_blocks(self):
        req=train_req(video_role='goal',depth_first_frame_conditioned=False)
        clean,noisy=window(8),window(9)
        state=torch.randn(2,8)
        packed=run_packed(req,clean,noisy,state)
        for blk in (1,2):
            one=run_packed(TT.block_request(req,blk),clean,noisy,state)
            self.assertTrue(close(packed[1][:,(blk-1)*16:blk*16],one[1][:,(blk-1)*16:blk*16]))
            self.assertTrue(close(packed[2][:,:,blk],one[2][:,:,blk]))

class ModeInference(unittest.TestCase):
    def test_idm_api_and_fixed_conditions(self):
        m=H.Harness(config()).eval()
        w=WAMInference(m,checkpoint='test',action_stats_version='test',camera_transform='test',settings=GenerationSettings(num_inference_steps=2))
        b=H.batch(b=1)
        o=Observation(b['first_frame'][0],b['state'][0],0,0.,None)
        calls=capture_calls(m)
        result=w.generate_action_for_goal(o,None,'pick',b['video_frames'][0,:4])
        self.assertIsInstance(result,GoalActionOutput)
        self.assertFalse(hasattr(result,'forecast'))
        self.assertEqual(result.actions.shape,(16,8))
        torch.testing.assert_close(calls[0]['video'],calls[-1]['video'])
        self.assertGreater(float(calls[0]['t_depth'].min()),0)
        self.assertEqual(calls[0]['temporal'].video_role,'goal')
        fc=w.predict_outcome(o,None,'pick',result.actions)
        self.assertEqual(fc.kind,'action_conditioned')
        for goal in (b['video_frames'][0,:1],torch.full_like(b['video_frames'][0,:4],float('nan'))):
            with self.assertRaises(ValueError):
                w.generate_action_for_goal(o,None,'pick',goal)

    def test_two_blocks_goal_prefix_consistency(self):
        m=H.Harness(config()).eval()
        b=H.batch()
        inp=dict(first_frame=b['first_frame'],state=b['state'],vlm_input_messages=b['vlm_input_messages'],num_inference_steps=2,decode_outputs=False)
        torch.manual_seed(1)
        _,a=m.inference_step(**inp,condition_video_frames=b['video_frames'][:,:4])
        torch.manual_seed(1)
        _,aa=m.inference_step(**inp,num_blocks=2,condition_video_frames=b['video_frames'])
        torch.testing.assert_close(a,aa[:,:16])
        with self.assertRaises(ValueError):
            m.inference_step(**inp,condition_video_frames=b['video_frames'][:,:4],condition_actions=b['actions'][:,:16])

if __name__=='__main__':
    unittest.main(verbosity=2)
