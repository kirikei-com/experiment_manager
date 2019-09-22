import pathlib
import subprocess
import json
import datetime
import shutil

import pandas as pd


class Experiment(object):
    """実験結果(experiment)を司るクラス
    """
    def __init__(self, experiment_folder, experiment_name):
        self.path = pathlib.Path(experiment_folder, experiment_name)
        if not self.path.exists():
            self.path.mkdir()
        
    def all_run(self):
        """experiment以下にあるrunを全て取得するメソッド
        """
        run_numbers = self._all_run_numbers()
        all_runs = [Run(self, n) for n in run_numbers]
        
        return all_runs
    
    def _all_run_numbers(self):
        """保存された全てのrunの番号を取得してlist(int)で返す
        """
        run_folder_iter = self.path.glob('[0-9]*')
        return [int(p.name.split('_')[0]) for p in run_folder_iter]
    
    def _new_run_number(self):
        """新しく発行されるrunの番号を返す
        """
        all_number_list = self._all_run_numbers()
        if len(all_number_list) > 0:
            return max(all_number_list) + 1  
        else:
            return 0 
    
    def start_logging(self, prefix_list=[]):
        return Run(self, self._new_run_number(), prefix_list, from_exp=True) 
    
    def summary(self, file_name):
        all_runs = self.all_run()
        
        all_df_list = []
        for r in all_runs:
            log = r.get_log(file_name)
            if not log is None:
                number = r.run_number
                df = pd.DataFrame.from_dict(log)
                df.columns = ['{}_{}'.format(number, m) for m in df.columns]
                all_df_list.append(df)
        
        if len(all_df_list) > 0:
            return pd.concat(all_df_list, axis=1)
        else:
            raise ValueError('Cannot find any log file: {}'.format(file_name))
        
class Run(object):
    """実験結果(run)を司るクラス
    """
    def __init__(self, experiment, run_number, prefix_list=[], from_exp=False):
        """フォルダを作成する
        """
        self.experiment = experiment
        self.run_number = run_number
        
        self.path = self._search_run_folder()
        
        if self.path is None:
            
            if from_exp:
                folder_name = self._generate_folder_name(prefix_list)
                self.path = pathlib.Path(self.experiment.path, folder_name)
                self.path.mkdir()
        
            else:
                raise ValueError('cannot find run number = {}'.format(run_number))
    
    def _search_run_folder(self):
        """指定したRunのフォルダをlistとして返す
            存在しなければNoneを返す
        """
        folder_list = list(self.experiment.path.glob('{}*'.format(self.run_number)))
        if len(folder_list) > 0:
            return folder_list[0]
        else:
            None
        
    def _generate_folder_name(self, prefix_list):
        """prefix_listとrun_numberを繋げたフォルダ名を作成
        """
        folder_name = str(self.run_number)
        for p in prefix_list:
            folder_name += '_{}'.format(p)
        return folder_name
    
    def delete(self):
        shutil.rmtree(str(self.path))
        
    def git_commit(self, prefix='run'):
        now = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
        subprocess.call(['git', 'add', '.'])
        subprocess.call(['git', 'commit', '-m', '"{}_{}"'.format(prefix, now)])
        commit_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])[:-1]
        print('git commit hash: {}'.format(commit_hash))
        return commit_hash.decode('utf-8') # 返り値がbyte型
        
    def save(self, contents, file_name, mode):
        file_path = self.path.joinpath(file_name)
        if mode == 'json':
            with open(file_path, mode='w') as f:
                json.dump(contents, f)
        
        elif mode == 'pickle':
            pd.to_pickle(contents, file_path)
            
    def get_log(self, file_name):
        log_path = self.path.joinpath(file_name)
        if log_path.exists():
            return pd.read_pickle(log_path)
        else:
            return None
     