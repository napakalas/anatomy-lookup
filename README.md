# anatomy-lookup

Ontology lookup to UBERON and ILX

### Install

Install from the latest release, for example:

```
pip install https://github.com/napakalas/anatomy-lookup/releases/download/v0.0.8/anatomy_lookup-0.0.8-py3-none-any.whl
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
al.search_with_scope('C1', ['Spinal cord'])
```

results

```
[('UBERON:0006469', 'C1 segment of cervical spinal cord', 0.9242348670959473),
 ('UBERON:0002240', 'spinal cord', 0.9151925444602966),
 ('UBERON:0003099', 'cranial neural crest', 0.8896005749702454),
 ('ILX:0794592', 'C1 spinal nerve', 0.886608362197876),
 ('UBERON:0005844', 'spinal cord segment', 0.8802721500396729)]
```

running with force is also available

```
al.search_with_scope('C1', ['Spinal cord'], force=True)
```

### Close instance to free resource

```python
al.close()
```

### Rebuild term embedding

```python
al.build_indexes()
```

This will download the latest release of SCKAN from https://github.com/SciCrunch/NIF-Ontology/releases
an then build the index

### Running annotation

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
