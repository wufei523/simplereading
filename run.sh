SERVER_PORT="$PORT";
fi
echo port is $SERVER_PORT
python manage.py runserver 0.0.0.0:$SERVER_PORT