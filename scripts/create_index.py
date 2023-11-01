from anatomy_lookup import AnatomyLookup
import argparse

if __name__ == '__main__':
    parser = parser = argparse.ArgumentParser(description='Creating Anatomy Lookup index')
    parser.add_argument('--file', dest='file', default=None)
    args = parser.parse_args()
    
    lookup = AnatomyLookup()
    lookup.build_indexes(file_path=args.file)

    lookup.close()
