#import esm
import torch
import numpy as np

from training.proteina.evaluations.fold_models.model import FoldModel
from transformers import AutoTokenizer, EsmForProteinFolding


class ESMFold(FoldModel):

	def __init__(self, device="cuda:0"):
		tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
		self.model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
		#self.model = esm.pretrained.esmfold_v1()
		self.device = device
		self.model = self.model.eval().cuda(device=device)

	def predict(self, seq):
		with torch.no_grad():
			output = self.model.infer(seq)
			pdb_str = self.model.output_to_pdb(output)[0]
			pae = (output['aligned_confidence_probs'].cpu().numpy()[0] * np.arange(64)).mean(-1) * 31
			mask = output['atom37_atom_exists'].cpu().numpy()[0,:,1] == 1
		return pdb_str, pae[mask,:][:,mask]
