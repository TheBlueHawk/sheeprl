import gymnasium as gym
import numpy as np
from ocatari.core import OCAtari
from gymnasium import spaces


OBJ_SIZE = 2
NOISE_SCALE = 2
# ENUM perturbation cateogry
class Perturbation:
    NONE = 0
    NOISE = 1
    OCCLUSION = 2
    FALSE_POSITIVE = 3
    NOISE_OCCLUSION = 4
    NOISE_FALSE_POSITIVE = 5
    OCCLUSION_FALSE_POSITIVE = 6
    ALL = 7


class OCAtariWrapper(gym.Wrapper):
    """OCAtari Environment that behaves like a gymnasium environment and passes env_check. Based on RAM, the observation space is object-centric.
    More specifically it is a list position history informations of objects detected.
    The observation space is a vector where every object position history has a fixed place.
    If an object is not detected its information entries are set to 0.
    """

    def __init__(self, id: str, render_mode:str, perturbation=Perturbation.NONE) -> None:
        # Assault works if hud is False, but not if it is True
        # Assault list of objects with hud True only has hud objects no game objects
        # workaround: set hud based on game name
        if id == "AssaultNoFrameskip-v4":
            hud = False
        else:
            hud = True
        self.ocatari_env = OCAtari(env_name=id, mode="revised", hud=hud, obs_mode="ori", render_mode=render_mode)
        super().__init__(self.ocatari_env)
        self.reference_list = self._init_ref_vector()
        self.current_vector = np.zeros(OBJ_SIZE * len(self.reference_list), dtype=np.uint8)
        if perturbation not in range(8):
            raise ValueError(f"Invalid perturbation {self.perturbation}\n Choose 0 for no perturbation, 1 for noise, 2 for occlusion, 3 for false positive.")
        else:
            self.perturbation = perturbation

    @property
    def observation_space(self):
        # fix to include pixel observations
        vl = len(self.reference_list) * OBJ_SIZE
        return spaces.Dict({
            "rgb": self.ocatari_env.observation_space,
            "objects_position": spaces.Box(low=0, high=255, shape=(vl,), dtype=np.uint8)
        })

    @property
    def action_space(self):
        return self.ocatari_env.action_space

    def step(self, *args, **kwargs):
        obs, reward, truncated, terminated, info = self.ocatari_env.step(*args, **kwargs)
        converted_obs = self._convert_obs(obs)
        return converted_obs, reward, truncated, terminated, info

    def reset(self, *args, **kwargs):
        obs, info = self.ocatari_env.reset(*args, **kwargs)
        converted_obs = self._convert_obs(obs)
        return converted_obs, info
    
    def _convert_obs(self, rgb_obs):
        self._obj2vec()
        obj = self.current_vector
        if self.perturbation == Perturbation.NONE:
            pass
        if self.perturbation == Perturbation.NOISE or self.perturbation == Perturbation.NOISE_OCCLUSION or self.perturbation == Perturbation.NOISE_FALSE_POSITIVE or self.perturbation == Perturbation.ALL:
            # add noise to the object position
            noise = np.random.normal(0, NOISE_SCALE, obj.shape)
            noise = np.where(obj == 0, 0, noise)
            # add noise cast to int only to the non-zero element of obj
            obj = [o + n for o, n in zip(obj, noise)]
            # clip the values to be in the range [0, 255]
            obj = np.clip(obj, 0, 255) 
            # back to uint8
            obj = np.array(obj, dtype=np.uint8)
        if self.perturbation == Perturbation.OCCLUSION or self.perturbation == Perturbation.NOISE_OCCLUSION or self.perturbation == Perturbation.OCCLUSION_FALSE_POSITIVE or self.perturbation == Perturbation.ALL:
            # zero out the object position of a few objects
            num_occlusions = np.random.randint(0, len(obj) // (OBJ_SIZE*3))
            occluded_indices = np.random.choice(len(obj) // OBJ_SIZE, num_occlusions, replace=False)
            for i in occluded_indices:
                obj[i * OBJ_SIZE : i * OBJ_SIZE + OBJ_SIZE] = 0
        if self.perturbation == Perturbation.FALSE_POSITIVE or self.perturbation == Perturbation.NOISE_FALSE_POSITIVE or self.perturbation == Perturbation.OCCLUSION_FALSE_POSITIVE or self.perturbation == Perturbation.ALL:
            # add a false positive object: look for two consecutive 0's and replace them with a random object position
            for i in range(len(obj) - 1):
                if obj[i] == 0 and obj[i + 1] == 0 and i % 2 == 0:
                    obj[i] = np.random.randint(0, 255)
                    obj[i + 1] = np.random.randint(0, 255)
                    break
        return {
            "rgb": rgb_obs,
            "objects_position": obj
        }

    def render(self, *args, **kwargs):
        return self.ocatari_env.render(*args, **kwargs)

    def close(self, *args, **kwargs):
        return self.ocatari_env.close(*args, **kwargs)

    def _init_ref_vector(self):
        reference_list = []
        obj_counter = {}       
        for o in self.ocatari_env.max_objects:
            if o.category not in obj_counter.keys():
                obj_counter[o.category] = 0
            obj_counter[o.category] += 1
        for k in list(obj_counter.keys()):
            reference_list.extend([k for i in range(obj_counter[k])])
        return reference_list

    def _obj2vec(self):
        temp_ref_list = self.reference_list.copy()
        for o in self.ocatari_env.objects:  # populate out_vector with object instance
            if o.category not in temp_ref_list:
                continue
            idx = temp_ref_list.index(o.category)  # at position of first category occurance
            start = idx * OBJ_SIZE
            flat = [item for item in o.xy]
            self.current_vector[start : start + OBJ_SIZE] = flat  # write the slice
            temp_ref_list[idx] = ""  # remove reference from reference list
        for i, d in enumerate(temp_ref_list):
            if d != "":  # fill not populated category instances wiht 0's
                self.current_vector[i * OBJ_SIZE : i * OBJ_SIZE + OBJ_SIZE] = [0]*OBJ_SIZE
