from DatabaseManagement import *
#record_operation('2021-08-25','TCEHY',0,300)
#add_security('TCEHY')
for i in tqdm(securities.keys()):
    update_data(i)
