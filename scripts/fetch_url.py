"""
Bollywood Songs YouTube URL Fetcher
This script searches for YouTube URLs for all the Bollywood songs and saves them to a CSV file.
"""

import csv
import time
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# You need to get your own API key from Google Cloud Console
# https://console.cloud.google.com/apis/credentials
API_KEY = 'work_done'

# Songs data organized by category
SONGS_DATA = {
    'ROMANTIC': [
        ('Tum Hi Ho', 'Aashiqui 2'),
        ('Raabta', 'Agent Vinod'),
        ('Tera Ban Jaunga', 'Kabir Singh'),
        ('Raanjhanaa Title Track', 'Raanjhanaa'),
        ('Samjhawan', 'Humpty Sharma Ki Dulhania'),
        ('Enna Sona', 'OK Jaanu'),
        ('Janam Janam', 'Dilwale'),
        ('Sun Saathiya', 'ABCD 2'),
        ('Tujh Mein Rab Dikhta Hai', 'Rab Ne Bana Di Jodi'),
        ('Kaun Tujhe', 'M.S. Dhoni'),
        ('Tum Se Hi', 'Jab We Met'),
        ('Pehli Nazar Mein', 'Race'),
        ('Tera Hone Laga Hoon', 'Ajab Prem Ki Ghazab Kahani'),
        ('Hawayein', 'Jab Harry Met Sejal'),
        ('Tere Sang Yaara', 'Rustom'),
        ('Hasi Ban Gaye', 'Hamari Adhuri Kahani'),
        ('Tum Mile (Title Track)', 'Tum Mile'),
        ('Dil Diyan Gallan', 'Tiger Zinda Hai'),
        ('Tujhe Kitna Chahne Lage', 'Kabir Singh'),
        ('Aaj Din Chadheya', 'Love Aaj Kal'),
        ('Agar Tum Mil Jao', 'Zeher'),
        ('Khuda Jaane', 'Bachna Ae Haseeno'),
        ('Tera Yaar Hoon Main', 'Sonu Ke Titu Ki Sweety'),
        ('Shiddat Title Track', 'Shiddat'),
        ('Raataan Lambiyan', 'Shershaah'),
        ('O Saathi', 'Baaghi 2'),
        ('Jeene Laga Hoon', 'Ramaiya Vastavaiya'),
        ('Gerua', 'Dilwale'),
        ('Tum Jo Aaye', 'Once Upon A Time In Mumbaai'),
        ('Main Rang Sharbaton Ka', 'Phata Poster Nikla Hero'),
    ],
    'SAD': [
        ('Agar Tum Saath Ho', 'Tamasha'),
        ('Channa Mereya', 'Ae Dil Hai Mushkil'),
        ('Tujhe Bhula Diya', 'Anjaana Anjaani'),
        ('Tadap Tadap', 'Hum Dil De Chuke Sanam'),
        ('Bhula Dena', 'Aashiqui 2'),
        ('Judaai', 'Badlapur'),
        ('Phir Bhi Tumko Chaahunga', 'Half Girlfriend'),
        ('Abhi Mujh Mein Kahin', 'Agneepath'),
        ('Humdard', 'Ek Villain'),
        ('Kabira (Encore)', 'Yeh Jawaani Hai Deewani'),
        ('Chal Wahan Jaate Hain', 'Arijit Singh'),
        ('Bekhayali', 'Kabir Singh'),
        ('Dard Dilo Ke', 'Xpose'),
        ('Tera Zikr', 'Darshan Raval'),
        ('Ae Dil Hai Mushkil (Title Track)', 'Ae Dil Hai Mushkil'),
        ('Khairiyat', 'Chhichhore'),
        ('Tu Hi Hai', 'Dear Zindagi'),
        ('Mann Mera', 'Table No. 21'),
        ('Main Dhoondne Ko Zamaane Mein', 'Heartless'),
        ('Lo Maan Liya', 'Raaz Reboot'),
        ('Hamari Adhuri Kahani', 'Title Track'),
        ('Dil Kyun Yeh Mera', 'Kites'),
        ('Tu Jaane Na', 'Ajab Prem Ki Ghazab Kahani'),
        ('Tujhe Sochta Hoon', 'Jannat 2'),
        ('Zindagi Kuch Toh Bata', 'Bajrangi Bhaijaan'),
        ('Pachtaoge', 'Arijit Singh'),
        ('Chitthi', 'Jubin Nautiyal'),
        ('Tum Ho', 'Rockstar'),
        ('Kun Faya Kun', 'Rockstar'),
        ('Jaan Nisar', 'Kedarnath'),
    ],
    'CALM': [
        ('Kun Faya Kun', 'Rockstar'),
        ('Ilahi', 'Yeh Jawaani Hai Deewani'),
        ('Khwaja Mere Khwaja', 'Jodhaa Akbar'),
        ('Maula Mere Maula', 'Anwar'),
        ('Nadaan Parindey', 'Rockstar'),
        ('Aayat', 'Bajirao Mastani'),
        ('Alvida', 'Life in a Metro'),
        ('Khaabon Ke Parinday', 'Zindagi Na Milegi Dobara'),
        ('Safarnama', 'Tamasha'),
        ('Tera Zikr', 'Darshan Raval'),
        ('Shikayatein', 'Lootera'),
        ('Saiyyara', 'Ek Tha Tiger'),
        ('Ye Honsla', 'Dor'),
        ('Kabira (Original)', 'YJHD'),
        ('Dil Mere', 'The Local Train'),
        ('Choo Lo', 'The Local Train'),
        ('Aashayein', 'Iqbal'),
        ('Manzar Hai Yeh Naya', 'Life in a Metro'),
        ('Subhanallah', 'Yeh Jawaani Hai Deewani'),
        ('Tu Aashiqui Hai', 'Jhankaar Beats'),
        ('Iktara', 'Wake Up Sid'),
        ('Aahista Aahista', 'Bachna Ae Haseeno'),
        ('Aas Paas Khuda', 'Anjaana Anjaani'),
        ('Behti Hawa Sa Tha Woh', '3 Idiots'),
        ('Phir Se Ud Chala', 'Rockstar'),
        ('Rehna Tu', 'Delhi-6'),
        ('Kabhi Kabhi Aditi', 'Jaane Tu Ya Jaane Na'),
        ('Maula Mere Le Le Meri Jaan', 'Chak De India'),
        ('Sajde', 'Kill Dil'),
        ('Dil Dhadakne Do', 'ZNMD'),
    ],
    'DRIVE': [
        ('Dil Chahta Hai', 'Title Track'),
        ('Khaabon Ke Parinday', 'ZNMD'),
        ('Ilahi', 'Yeh Jawaani Hai Deewani'),
        ('Sooraj Dooba Hai', 'Roy'),
        ('Zinda', 'Bhaag Milkha Bhaag'),
        ('Safarnama', 'Tamasha'),
        ('Hairat', 'Anjaana Anjaani'),
        ('Phir Se Ud Chala', 'Rockstar'),
        ('Patakha Guddi', 'Highway'),
        ('Aashayein', 'Iqbal'),
        ('Allah Ke Bande', 'Kailash Kher'),
        ('Aao Milo Chalein', 'Jab We Met'),
        ('Uff Teri Ada', 'Karthik Calling Karthik'),
        ('Ik Junoon (Paint It Red)', 'ZNMD'),
        ('Zinda Hoon Yaar', 'Lootera'),
        ('Chal Chalein', 'Lakshya'),
        ('Dil Dhadakne Do', 'ZNMD'),
        ('Ye Dooriyan', 'Love Aaj Kal'),
        ('Shaam', 'Aisha'),
        ('Behti Hawa Sa Tha Woh', '3 Idiots'),
        ('Kholo Kholo', 'Taare Zameen Par'),
        ('Tum Se Hi (Reprise)', 'Jab We Met'),
        ('Tera Hone Laga Hoon (Reprise)', 'Ajab Prem Ki Ghazab Kahani'),
        ('Jeene Ke Hai Chaar Din', 'Mujhse Shaadi Karogi'),
        ('Gallan Goodiyan', 'Dil Dhadakne Do'),
        ('Yeh Ishq Hai', 'Jab We Met'),
        ('Hum Jo Chalne Lage', 'Jab We Met'),
        ('Main Hoon Na (Title)', 'Main Hoon Na'),
        ('Kyon', 'Barfi!'),
        ('Mitwa', 'Kabhi Alvida Naa Kehna'),
    ],
    'PARTY': [
        ('Kala Chashma', 'Baar Baar Dekho'),
        ('Kar Gayi Chull', 'Kapoor & Sons'),
        ('Bom Diggy Diggy', 'Sonu Ke Titu Ki Sweety'),
        ('London Thumakda', 'Queen'),
        ('Nashe Si Chadh Gayi', 'Befikre'),
        ('Badtameez Dil', 'YJHD'),
        ('The Breakup Song', 'Ae Dil Hai Mushkil'),
        ('Malhari', 'Bajirao Mastani'),
        ('Aankh Marey', 'Simmba'),
        ('Dil Dhadakne Do', 'ZNMD'),
        ('Gallan Goodiyan', 'DDD'),
        ('Desi Girl', 'Dostana'),
        ('Saturday Saturday', 'Humpty Sharma Ki Dulhania'),
        ('Subha Hone Na De', 'Desi Boyz'),
        ('Party On My Mind', 'Race 2'),
        ('Baby Ko Bass Pasand Hai', 'Sultan'),
        ('Abhi Toh Party Shuru Hui Hai', 'Khoobsurat'),
        ('Sweety Tera Drama', 'Bareilly Ki Barfi'),
        ('Proper Patola', 'Namaste England'),
        ('Dil Chori', 'SKTKS'),
        ('Tera Hero Idhar Hai', 'Main Tera Hero'),
        ('Hookah Bar', 'Khiladi 786'),
        ('Tumhi Ho Bandhu', 'Cocktail'),
        ("Let's Nacho", 'Kapoor & Sons'),
        ('High Heels', 'Ki & Ka'),
        ('Cutiepie', 'Ae Dil Hai Mushkil'),
        ('Suraj Dooba Hai', 'Roy'),
        ('Naacho Naacho', 'RRR'),
        ('Jee Karda', 'Singh Is Kinng'),
        ('Main Tera Boyfriend', 'Raabta'),
    ]
}


def search_youtube_video(youtube, song_name, movie_name):
    """
    Search for a YouTube video and return the first result's URL
    """
    try:
        # Create search query
        query = f"{song_name} {movie_name} official"
        
        # Call YouTube API
        search_response = youtube.search().list(
            q=query,
            part='id,snippet',
            maxResults=1,
            type='video'
        ).execute()
        
        # Extract video ID and create URL
        if search_response['items']:
            video_id = search_response['items'][0]['id']['videoId']
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            video_title = search_response['items'][0]['snippet']['title']
            return video_url, video_title
        else:
            return None, None
            
    except HttpError as e:
        print(f"An HTTP error occurred: {e}")
        return None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None


def fetch_all_urls():
    """
    Fetch YouTube URLs for all songs and save to CSV
    """
    # Initialize YouTube API client
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    
    # Prepare CSV file
    output_file = 'bollywood_songs_with_urls.csv'
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Category', 'Song Name', 'Movie', 'YouTube URL', 'Video Title', 'Status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        total_songs = sum(len(songs) for songs in SONGS_DATA.values())
        current_song = 0
        
        # Process each category
        for category, songs in SONGS_DATA.items():
            print(f"\n{'='*60}")
            print(f"Processing {category} songs...")
            print(f"{'='*60}")
            
            for song_name, movie_name in songs:
                current_song += 1
                print(f"\n[{current_song}/{total_songs}] Searching: {song_name} - {movie_name}")
                
                # Search for the song
                url, video_title = search_youtube_video(youtube, song_name, movie_name)
                
                # Write to CSV
                if url:
                    print(f"✓ Found: {url}")
                    writer.writerow({
                        'Category': category,
                        'Song Name': song_name,
                        'Movie': movie_name,
                        'YouTube URL': url,
                        'Video Title': video_title,
                        'Status': 'Found'
                    })
                else:
                    print(f"✗ Not found")
                    writer.writerow({
                        'Category': category,
                        'Song Name': song_name,
                        'Movie': movie_name,
                        'YouTube URL': '',
                        'Video Title': '',
                        'Status': 'Not Found'
                    })
                
                # Add delay to respect API rate limits
                time.sleep(0.5)
    
    print(f"\n{'='*60}")
    print(f"Complete! Results saved to: {output_file}")
    print(f"{'='*60}")


def main():
    """
    Main function
    """
    print("="*60)
    print("Bollywood Songs YouTube URL Fetcher")
    print("="*60)
    print(f"\nTotal songs to process: {sum(len(songs) for songs in SONGS_DATA.values())}")
    print("\nIMPORTANT: Make sure you have:")
    print("1. Installed required package: pip install google-api-python-client")
    print("2. Added your YouTube API key in the script")
    print("\nStarting in 3 seconds...")
    time.sleep(3)
    
    if API_KEY == 'YOUR_YOUTUBE_API_KEY_HERE':
        print("\n❌ ERROR: Please add your YouTube API key!")
        print("\nTo get an API key:")
        print("1. Go to https://console.cloud.google.com/")
        print("2. Create a new project or select existing one")
        print("3. Enable YouTube Data API v3")
        print("4. Create credentials (API Key)")
        print("5. Copy the API key and paste it in the script")
        return
    
    fetch_all_urls()


if __name__ == "__main__":
    main()