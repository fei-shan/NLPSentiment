import os
PROJECT_DIR = '.'
LOCAL_DATA_DIR = os.path.join(PROJECT_DIR, 'data')
IMDB_50K_CSV = os.path.join(LOCAL_DATA_DIR, 'IMDB Dataset.csv')
# IMDB_DATA_DIR = os.path.join(LOCAL_DATA_DIR, 'aclImdb')


STEREOSET_TERMS = {
    'gender': [['man', 'son', 'father', 'boy', 'brother', 'grandfather', 'husband', 'schoolboy', 'gentleman', 'himself',
                'daddy', 'groom', 'male', 'gentlemen', 'he', 'stepfather', 'boyfriend', 'him', 'his', 'sir'],
               ['woman', 'daughter', 'mother', 'girl', 'sister', 'grandmother', 'wife', 'schoolgirl', 'lady', 'herself',
                'mommy', 'bride', 'female', 'ladies', 'she', 'stepmother', 'girlfriend', 'her', 'hers', 'ma_am']],  # male-female
    'profession': [['barber'], ['coach'], ['businessperson'], ['football_player'], ['construction_worker'],
                   ['manager'], ['ceo'], ['accountant'], ['commander'], ['firefighter'], ['mover'],
                   ['software_developer'], ['guard'], ['baker'], ['doctor'], ['athlete'], ['artist'], ['dancer'],
                   ['mathematician'], ['janitor'], ['carpenter'], ['mechanic'], ['actor'], ['handyman'],
                   ['musician'], ['detective'], ['politician'], ['entrepreneur'], ['model'], ['opera_singer'],
                   ['chief'], ['lawyer'], ['farmer'], ['writer'], ['librarian'], ['army'], ['real-estate_developer'],
                   ['broker'], ['scientist'], ['butcher'], ['electrician'], ['prosecutor'], ['banker'],
                   ['cook'], ['hairdresser'], ['prisoner'], ['plumber'], ['attorney'], ['boxer'], ['chess_player'],
                   ['priest'], ['swimmer'], ['tennis_player'], ['supervisor'], ['attendant'],
                   ['housekeeper'], ['maid'], ['producer'], ['researcher'], ['midwife'], ['judge'], ['umpire'],
                   ['bartender'], ['economist'], ['physicist'], ['psychologist'], ['theologian'], ['salesperson'],
                   ['physician'], ['sheriff'], ['cashier'], ['assistant'], ['receptionist'], ['editor'], ['engineer'],
                   ['comedian'], ['painter'], ['civil servant'], ['diplomat'], ['guitarist'], ['linguist'],
                   ['poet'], ['laborer'], ['teacher'], ['delivery_man'], ['realtor'], ['pilot'], ['professor'],
                   ['chemist'], ['historian'], ['pensioner'], ['performing_artist'], ['singer'], ['secretary'],
                   ['auditor'], ['counselor'], ['designer'], ['soldier'], ['journalist'], ['dentist'], ['analyst'],
                   ['nurse'], ['tailor'], ['waiter'], ['author'], ['architect'], ['academic'], ['director'],
                   ['illustrator'], ['clerk'], ['policeman'], ['chef'], ['photographer'], ['drawer'], ['cleaner'],
                   ['pharmacist'], ['pianist'], ['composer'], ['handball_player'], ['sociologist']],  # uncategorized
    'race': [['nigerian'], ['ethiopia'], ['europe'], ['european'], ['russian'], ['ukraine'], ['sudan'], ['afghanistan'],
             ['iraq'], ['yemen'], ['ukrainian'], ['russia'], ['italy'], ['somali'], ['iran'], ['afghan'], ['indian'],
             ['italian'], ['australian'], ['spanish'], ['guatemalan'], ['hispanic'], ['venezuela'], ['sudanese'],
             ['oman'], ['finnish'], ['swedish'], ['venezuelan'], ['puerto_rican'], ['ghanaian'], ['moroccan'],
             ['somalia'], ['saudi_arabian'], ['syria'], ['chinese'], ['pakistani'], ['china'], ['india'],
             ['irish'], ['britain'], ['france'], ['greece'], ['scotland'], ['mexican'], ['paraguayan'], ['brazil'],
             ['african'], ['eritrean'], ['sierra_leonean'], ['africa'], ['jordan'], ['indonesia'], ['vietnam'],
             ['pakistan'], ['german'], ['romania'], ['brazilian'], ['ecuadorian'], ['mexico'], ['puerto_rico'], ['kenyan'],
             ['liberian'], ['cameroonian'], ['african_americans'], ['kenya'], ['liberia'], ['sierra_leon'],
             ['qatari'], ['syrian'], ['arab'], ['saudi_arabia'], ['lebanon'], ['indonesian'], ['french'],
             ['norweigan'], ['south_africa'], ['jordanian'], ['korea'], ['singapore'], ['romanian'], ['crimean'],
             ['native_american'], ['germany'], ['ireland'], ['ecuador'], ['morocco'], ['omani'], ['iranian'],
             ['iraqi'], ['qatar'], ['turkey'], ['vietnamese'], ['nepali'], ['laos'], ['bangladesh'], ['british'],
             ['polish'], ['greek'], ['scottish'], ['bolivian'], ['guatemala'], ['ghana'], ['cameroon'], ['japanese'],
             ['taiwanese'], ['bengali'], ['nepal'], ['albanian'], ['albania'], ['columbian'], ['peruvian'],
             ['argentian'], ['spain'], ['paraguay'], ['ethiopian'], ['egyptian'], ['persian_people'], ['sweden'],
             ['crimea'], ['portuguese'], ['argentina'], ['chile'], ['cape_verdean'], ['turkish'], ['yemeni'],
             ['taiwan'], ['austrian'], ['white_people'], ['finland'], ['australia'], ['south_african'],
             ['eriteria'], ['egypt'], ['korean'], ['dutch_people'], ['peru'], ['poland'], ['chilean'],
             ['columbia'], ['bolivia'], ['laotian'], ['lebanese'], ['japan'], ['norway'], ['cape_verde'],
             ['portugal'], ['austria'], ['singaporean'], ['netherlands']],  # uncategorized
    'religion': [['islam', 'sharia', 'jihad', 'muslim', 'mohammed', 'quran'],
                 ['christian', 'church', 'bible', 'holy_trinity'],
                 ['hindu', 'brahmin']],  # islam-christian-hindu
}

# from local_config import *

