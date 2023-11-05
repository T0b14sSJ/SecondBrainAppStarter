# Second Brain App

This is a chatbot-like application that allows you to store and retrieve notes, ideas, tasks, and schedules. It uses OpenAI's GPT-4 model and a vector database to store and retrieve information.

## Features

- Chat interface powered by Streamlit
- Store information in a file and index it to a vector database
- Retrieve information from the vector database

## Setup

To use this application, you need to have the following:

- OpenAI API key
- OpenAI API embedding model
- OpenAI model

These should be stored in your Streamlit secrets.

## Functions

The application has two main functions:

- `store_information(information)`: This function stores the provided information into a file and indexes it to a vector database.
- `get_information_from_memory_db(information)`: This function retrieves information from the vector database based on the provided query.

## Usage

To use the application, simply run the script and interact with the chat interface. You can ask the bot to store information or retrieve information from the database.

## Improvements

Currently, the application stores data in a local file and the vector database is also stored locally. Future improvements could include storing the data and the vector database in the cloud or in a SQL database for more robust and scalable storage.

## Note

This application is a proof of concept and is not intended for production use. It is a demonstration of how to use OpenAI's GPT-4 model and a vector database to create a simple chatbot that can store and retrieve information.