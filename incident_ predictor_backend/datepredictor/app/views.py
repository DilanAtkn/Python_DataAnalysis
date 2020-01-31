from rest_framework.viewsets import ModelViewSet

from app.models import Prediction, PredictData
from app.serializers import PredictionSerializer, PredictDataSerializer


class PredictionView(ModelViewSet):
    queryset = PredictData.objects.all()
    serializer_class = PredictDataSerializer
    first = True

    def get_queryset(self):
        if self.request.query_params.get('incident_state','') and PredictionView.first is True:
            incident_state = str(self.request.query_params.get('incident_state', '')).capitalize()
            reassignment_count = self.request.query_params.get('reassignment_count', '')
            reopen_count = self.request.query_params.get('reopen_count', '')
            sys_mod_count = self.request.query_params.get('sys_mod_count', '')
            contact_type = str(self.request.query_params.get('contact_type', '')).capitalize()
            location = self.request.query_params.get('location', '')
            category = self.request.query_params.get('category', '')
            subcategory = self.request.query_params.get('subcategory', '')
            priority = self.request.query_params.get('priority', '')
            Prediction.objects.create(incident_state=incident_state, reassignment_count=reassignment_count, reopen_count=reopen_count,
                                      sys_mod_count=sys_mod_count,contact_type=contact_type,location=location, priority=priority,
                                      category=category,subcategory=subcategory)

            #------------------------------------------------------------------------
            import pandas as pd
            import numpy as np

            from sklearn.preprocessing import LabelEncoder
            import pickle
            import warnings
            warnings.filterwarnings("ignore")
            import os
            csv_file = os.path.join(r'C:\Users\dilan\OneDrive\Documents\incident_ predictor_backend\datepredictor', 'incident_event_log.csv')
            print(csv_file)
            df = pd.read_csv(csv_file)
            # removing extra information from feature values

            df['caller_id'] = df['caller_id'].apply(lambda x: str(x).split(' ')[-1] if x != np.nan else np.nan)
            df['opened_by'] = df['opened_by'].apply(lambda x: str(x).split(' ')[-1] if x != np.nan else np.nan)
            df['closed_code'] = df['closed_code'].apply(lambda x: str(x).split(' ')[-1] if x != np.nan else np.nan)
            df['resolved_by'] = df['resolved_by'].apply(lambda x: str(x).split(' ')[-1] if x != np.nan else np.nan)
            df['sys_created_by'] = df['sys_created_by'].apply(
                lambda x: str(x).split(' ')[-1] if x != np.nan else np.nan)
            df['sys_updated_by'] = df['sys_updated_by'].apply(
                lambda x: str(x).split(' ')[-1] if x != np.nan else np.nan)
            df['location'] = df['location'].apply(lambda x: str(x).split(' ')[-1] if x != np.nan else np.nan)
            df['category'] = df['category'].apply(lambda x: str(x).split(' ')[-1] if x != np.nan else np.nan)
            df['subcategory'] = df['subcategory'].apply(lambda x: str(x).split(' ')[-1] if x != np.nan else np.nan)
            df['u_symptom'] = df['u_symptom'].apply(lambda x: str(x).split(' ')[-1] if x != np.nan else np.nan)
            df['cmdb_ci'] = df['cmdb_ci'].apply(lambda x: str(x).split(' ')[-1] if x != np.nan else np.nan)
            df['impact'] = df['impact'].apply(lambda x: str(x).split(' ')[0] if x != np.nan else np.nan)
            df['urgency'] = df['urgency'].apply(lambda x: str(x).split(' ')[0] if x != np.nan else np.nan)
            df['priority'] = df['priority'].apply(lambda x: str(x).split(' ')[0] if x != np.nan else np.nan)
            df['assignment_group'] = df['assignment_group'].apply(
                lambda x: str(x).split(' ')[-1] if x != np.nan else np.nan)
            df['assigned_to'] = df['assigned_to'].apply(lambda x: str(x).split(' ')[-1] if x != np.nan else np.nan)
            df['problem_id'] = df['problem_id'].apply(lambda x: str(x).split(' ')[-1] if x != np.nan else np.nan)
            df['vendor'] = df['vendor'].apply(lambda x: str(x).split(' ')[-1] if x != np.nan else np.nan)
            df['caused_by'] = df['caused_by'].apply(lambda x: str(x).split(' ')[-1] if x != np.nan else np.nan)
            df['closed_code'] = df['closed_code'].apply(lambda x: str(x).split(' ')[-1] if x != np.nan else np.nan)
            df['opened_at'] = df['opened_at'].apply(lambda x: str(x).split(' ')[0] if x != np.nan else np.nan)
            df['resolved_at'] = df['resolved_at'].apply(lambda x: str(x).split(' ')[0] if x != np.nan else np.nan)
            df['closed_at'] = df['closed_at'].apply(lambda x: str(x).split(' ')[0] if x != np.nan else np.nan)

            df.replace('?', np.nan, inplace=True)

            df['reassignment_count'] = pd.to_numeric(df["reassignment_count"])
            df['category'] = pd.to_numeric(df["category"])
            df['subcategory'] = pd.to_numeric(df["subcategory"])
            df['priority'] = pd.to_numeric(df["priority"])
            df['assignment_group'] = pd.to_numeric(df["assignment_group"])
            # **Notice**
            # - There are some dates in the dataset like opened at and closed at these must be in the **datetime** formet.

            # Lets convert its datatype from object to datetime

            df['opened_at'] = pd.to_datetime(df['opened_at'])
            df['closed_at'] = pd.to_datetime(df['closed_at'])
            df['sys_created_at'] = pd.to_datetime(df['sys_created_at'])
            df['sys_updated_at'] = pd.to_datetime(df['sys_updated_at'])
            df['resolved_at'] = pd.to_datetime(df['resolved_at'])

            # subtraction of closing anf opening date of an incident return us days used for closing that incident
            df['closed_at_opened_at'] = df['closed_at'] - df['opened_at']

            # Let's see instances with -ve values of closed_at_opened_at feature

            e = df[df.closed_at_opened_at < '0']

            p = e.index
            df = df.drop(p)

            df['incident_state'] = df['incident_state'].replace('-100', 'Active')


            df['location'] = df['location'].fillna('204')
            df['category'] = df['category'].fillna('53')
            df['subcategory'] = df['subcategory'].fillna('174')
            selected_df = df.copy()

            # In[66]:

            features_drop = ['number', 'active', 'made_sla', 'caller_id', 'opened_by', 'opened_at',
                             'sys_created_by', 'sys_created_at', 'sys_updated_by', 'sys_updated_at', 'u_symptom',
                             'cmdb_ci', 'impact', 'urgency', 'assignment_group',
                             'assigned_to', 'knowledge', 'u_priority_confirmation', 'notify',
                             'problem_id', 'rfc', 'vendor', 'caused_by', 'closed_code',
                             'resolved_by', 'resolved_at', 'closed_at', ]

            selected_df = selected_df.drop(features_drop, axis=1)

            selected_df['closed_at_opened_at'] = selected_df['closed_at_opened_at'].apply(
                lambda x: str(x).split(' ')[0] if x != np.nan else np.nan)

            #Convert the data type of column
            selected_df = selected_df.astype(
                {'category': 'int64', 'subcategory': 'int64', 'closed_at_opened_at': 'int64', 'location': 'int64'})

            labelEncoder_incident_state = LabelEncoder()
            labelEncoder_incident_state.fit(selected_df.incident_state)
            selected_df['incident_state'] = labelEncoder_incident_state.transform(selected_df.incident_state)

            labelEncoder_contact_type = LabelEncoder()
            labelEncoder_contact_type.fit(selected_df.contact_type)
            selected_df['contact_type'] = labelEncoder_contact_type.transform(selected_df.contact_type)
            import os
            import pickle
            file1 = os.path.join(r'C:\Users\dilan\OneDrive\Documents\incident_ predictor_backend\datepredictor\app','BestModel.pkl')
            with open(file1, 'rb') as pickle_file:
                bestmodel = pickle.load(pickle_file)
            

            print("----------model loaded")
            dictt = {'incident_state': [incident_state], 'reassignment_count': [reassignment_count],
                     'reopen_count': [reopen_count], 'sys_mod_count': [sys_mod_count], 'contact_type': [contact_type],
                     'location': [location], 'category': [category]
                , 'subcategory': [subcategory], 'priority': [priority]}

            input1 = pd.DataFrame(dictt)

            labelEncoder_incident_state.transform(input1.incident_state)
            labelEncoder_contact_type.transform(input1.contact_type)

            input1['incident_state'] = labelEncoder_incident_state.transform(input1.incident_state)
            input1['contact_type'] = labelEncoder_contact_type.transform(input1.contact_type)

            y_predict = bestmodel.predict(input1)

            #-------------------------------------------------------------------------

            obj = Prediction.objects.latest('id')
            PredictData.objects.create(prediction_fk_id=obj.id,predic=y_predict)
            return PredictData.objects.filter(prediction_fk_id=obj.id)

        else:
            print('else part get_queryset')
            PredictionView.first = True
            return PredictionView.objects.all()




class PredictDataView(ModelViewSet):
    queryset = PredictData.objects.all()
    serializer_class = PredictDataSerializer
