# anatomy-lookup
Ontology lookup to UBERON and ILX

### Install
```
pip install git+https://github.com/napakalas/anatomy-lookup.git
```

### Lookup use
```python
from anatomy_lookup import AnatomyLookup

al = AnatomyLookup()
al.search('nasophayrnx')
```
results
```
('http://purl.obolibrary.org/obo/UBERON_0001728',
 'nasopharynx',
 0.8313473463058472)
```

### Update UBERON and ILX terms
```python
AnatomyLookup.update_terms()
```

###  Rebuild term embedding
```python
al.build_indexes('path to ttl files')
```

ttl files can be obtained from https://github.com/SciCrunch/NIF-Ontology/releases