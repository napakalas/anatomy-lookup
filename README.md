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

can use force
```
al.search('cochlear g.', force=True)
```
results
```
('http://purl.obolibrary.org/obo/UBERON_0000395',
 'cochlear ganglion',
 1.0000001192092896)
```

### Update UBERON and ILX terms
```
al.update_terms()
```

### Lookup with scope
```
al.search_with_scope('1st plantar metatarsal', ['nerve','brain'])
```
results
```
[('http://purl.obolibrary.org/obo/UBERON_0035195',
  'plantar metatarsal artery',
  0.7471498250961304),
 ('http://purl.obolibrary.org/obo/UBERON_0003650',
  'metatarsal bone of digit 1',
  0.723603367805481),
 ('http://purl.obolibrary.org/obo/UBERON_0001448',
  'metatarsal bone',
  0.6906536817550659),
 ('http://purl.obolibrary.org/obo/UBERON_0003652',
  'metatarsal bone of digit 3',
  0.6834049224853516),
 ('http://purl.obolibrary.org/obo/UBERON_0003651',
  'metatarsal bone of digit 2',
  0.682817816734314)]
```
running with force is also available
```
lookup.search_with_scope('1st plantar metatarsal', ['nerve'], force=True)
```

### Close instance to free resource
```python
al.close()
```

###  Rebuild term embedding
```python
al.build_indexes()
```
This will download the latest release of SCKAN from https://github.com/SciCrunch/NIF-Ontology/releases
an then build the index

### Running annotaation
```
from anatomy_lookup import AnatomyAnnotator
anno = AnatomyAnnotator()
anno.annotate('annotation.json','name', ['systems', 'organ'])
```
can use force also
```
anno.annotate('annotation.json','name', ['systems', 'organ'], force=True)
```

### Save to xlsx
```
anno.save_to_xlsx('annotation_test.xlsx')
```

### Save to json
```
anno.save_to_json('annotation_test.json')
```
