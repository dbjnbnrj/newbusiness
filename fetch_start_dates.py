import os
import json
import pprint
from datetime import date
import dateutil.parser as dparser
import time

start=time.time()
initial_business_dataset_filename='yelp_academic_dataset_business.json'
initial_review_dataset_filename='yelp_academic_dataset_review.json'
locality="Las Vegas"
locality_business_filename=locality+'_business.json'
locality_business_start_dates_filename=locality+'_business_start_dates.json'

def readBusinessJson(read_filename, write_filename):
	f=open(read_filename,'r')
	f2=open(write_filename,'w')
	for line in f:
		business_json=json.loads(line)
		if business_json["city"]==locality:
			json.dump(business_json, f2)
			f2.write("\n")
	f.close()
	f2.close()



def extractReviewsJsonForALocation(businessFileName, reviewFileName, destinationFileName):
	#create dictionary with key business_id & value as date of first review
	f=open(businessFileName, 'r')
	f2=open(reviewFileName, 'r')
	business_date_dict={}
	for line in f:
		business_json=json.loads(line)
		business_date_dict[business_json['business_id']]=None
	f.close()
	for line in f2:
		review_json=json.loads(line)
		if(review_json["business_id"] in business_date_dict):
			date=dparser.parse(review_json['date'],fuzzy=True)
			if business_date_dict[review_json["business_id"]]==None:
				business_date_dict[review_json["business_id"]]=date
			elif(date< business_date_dict[review_json["business_id"]]):
					business_date_dict[review_json["business_id"]]=date
	f2.close()
	f3=open(destinationFileName, 'w')
	for key, value in business_date_dict.items():
		if value is None:
			continue
		else:
			print "{"+ key+" :"+value.strftime("%d-%m-%Y")+"}\n"
			f3.write("{ \""+ key+"\": \""+value.strftime("%d-%m-%Y")+"\" }\n")
	f3.close()

readBusinessJson(initial_business_dataset_filename, locality_business_filename)
extractReviewsJsonForALocation(locality_business_filename, initial_review_dataset_filename, locality_business_start_dates_filename)



print "the code took: "
print (time.time() - start)




