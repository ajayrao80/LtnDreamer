import ltn
from utils.utils import cosine_similarity
import torch

And = ltn.Connective(ltn.fuzzy_ops.AndProd())
Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2, stable=True), quantifier="f")
Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=2), quantifier="e")
SatAgg = ltn.fuzzy_ops.SatAgg()
Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
Sim = ltn.Predicate(func=lambda x, y: cosine_similarity(x, y))
Eq = ltn.Predicate(func=lambda x, y: torch.exp(-torch.norm(x - y, dim=1))) 