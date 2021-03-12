class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'pascal':
            return '/nethome/hkwon64/Research/imuTube/repos_v2/human_parsing/Grapy-ML/data/datasets/pascal/'  # folder that contains pascal/.
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError
