import requests
import json
import cv2 
import os

def determine_sex(url):
    if 'female' in url:
        sex = 'female'
    elif 'male' in url:
        sex = 'male'
    else:
        return None
    return sex

def determine_race(url):
    if 'asian' in url:
        race = 'asian'
    elif 'white' in url:
        race = 'white'
    elif 'black' in url:
        race = 'black'
    else:
        return None
    return race

def determine_part(url):
    if 'mouth' in url:
        part = 'mouth'
    elif 'eyes' in url:
        part = 'eyes'
    elif 'jaw' in url:
        part = 'jaw'
    elif 'nose' in url:
        part = 'nose'
    elif 'hair' in url:
        part = 'hair'
    else:
        return None
    return part

def download_img(url):
    dir_name = 'dataset/face_parts'
    # get img attributes.
    sex, race, part = determine_sex(url), determine_race(url), determine_part(url)

    if part is None or race is None or sex is None:
        return None

    # make dir.
    if not os.path.exists(os.path.join(dir_name,sex,race,part)):
        os.makedirs(os.path.join(dir_name,sex,race,part))

    # create filename.
    idx = len(os.listdir(os.path.join(dir_name,sex,race,part)))
    if part=='hair':
        index = str(idx) if idx > 9 else '0' + str(idx) 
    else:
        index = str(idx)
    filename = part + '_' + index +'.png'
    img_data = requests.get(url).content

    with open(os.path.join(dir_name,sex,race,part,filename), 'wb') as handler:
        handler.write(img_data)

def download_masks(url):
    dir_name = 'dataset/masks'
    files = [
        'files/assets/20728527/1/Face%20Pieces%20-%20eyes%20mask.png',
        'files/assets/20728528/1/Face%20Pieces%20-%20nose%20mask.png',
        'files/assets/20728527/1/Face%20Pieces%20-%20eyes%20mask.png'
    ]

    for file, filename in zip(files, ['eyes_mask_new.png', 'nose_mask_new.png', 'mouth_mask_new.png']):
        img_data = requests.get(os.path.join(url,file)).content

        with open(os.path.join(dir_name,filename), 'wb') as handler:
            handler.write(img_data)

        img = cv2.imread(os.path.join(dir_name,filename), cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (640,640)) 

        img = img[:,:,3]
        if 'mouth' not in filename:
            img = img/255.0

        cv2.imwrite(os.path.join(dir_name,filename), img)

def cleanup_imgs():
    """
    The following files are useless for our experiments:
    => FEMALE:
        - ASIAN: 6-11
        - BLACK: 6-9
        - WHITE: 6-11

        => MALE:
        - ASIAN: 0,2,4,7,8,10
        - BLACK: 0,2,4,6,8
        - WHITE: -
    """

    genders, races, part_name = ['female', 'male'], ['asian', 'black', 'white'], 'hair' 

    indices = [
        ['06', '07', '08', '09', '10', '11'], 
        ['06', '07', '08', '09'], ['06', '07', '08', '09', '10', '11'],
        ['00', '02', '04', '07', '08', '10'], ['00', '02', '04', '06', '08'], []
        ]

    counter = 0
    for gender in genders:
        for race in races:
            filenames = [f'{part_name}_{idx}.png' for idx in indices[counter]]
            counter+=1

            for filename in filenames:
                os.remove(os.path.join('./dataset/face_parts', gender, race, part_name, filename))


    counter = 0
    for gender in genders:
        for race in races:
            filenames = next(os.walk(f'./dataset/face_parts/{gender}/{race}/hair'), (None, None, []))[2]
            filenames.sort()

            for idx, filename in enumerate(filenames):
                os.rename(
                    os.path.join('./dataset/face_parts', gender, race, part_name, filename), 
                    os.path.join('./dataset/face_parts', gender, race, part_name, f'{part_name}_{idx}.png')
                    )

def download_data():
    print('=> Downloading part-images from the web..')
    # Load web-page config file.
    filename = './dataset/config.json'
    f = open(filename,)
    config = json.load(f)

    # Download images.
    url = 'http://www2.open.ac.uk/openlearn/photoFit-me2/Photofit/'

    for img in config['assets']:
        img_name = config['assets'][img]['name']
        img_dir  = config['assets'][img]['file']

        if img_dir is not None:
            img_url = url + img_dir['url'] + '?t=' + img_dir['hash'] 
            download_img(img_url)

    cleanup_imgs()