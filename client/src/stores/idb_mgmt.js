
import { openDB, deleteDB, wrap, unwrap } from 'idb'

import { getFormattedFileSize } from '@/utils/utils.js'
import { DocumentRecord } from './data.js'





export class IdbConfig{
    constructor(dbName, dbVersion, storeNamesAndKeyFields){
        this.dbName = dbName
        this.dbVersion = dbVersion
        this.storeNamesAndKeyFields = storeNamesAndKeyFields
        this.isSupported = null
    }

    async checkIdbSupport(){
        // Check for IndexedDB support
        //notes: Chrome now lets an origin use 60% of the storage device's space (Real nitty gritty: "storage device" is the partition containing the chrome profile directory).
        //ref: https://web.dev/articles/indexeddb
        if (!('indexedDB' in window)) {
          // Can't use IndexedDB
          console.log("ERROR: This browser doesn't support IndexedDB")
          this.isSupported = false
        } else {
          // Do IndexedDB stuff here:
          // ...
          console.log("This browser supports IndexedDB")
          this.isSupported = true
          await getIdbAvalabileSpace()
        }
        return this.isSupported
    }
    async testConfig(IdbConfig){
      if(true){
        const record = new DocumentRecord()
        record.id = 'test'
        const testData = [0,0,0,0,0]
        const result1 = await record.setDataArray(testData)
        const check1 = isNaN(result1[0])==false 
        console.log(check1) 

        const result2 = await record.getDataArray()
        const check2 = JSON.stringify(result2) == JSON.stringify(testData) 
        console.log(check2 )
      }
      if(true){
        const record = new DocumentRecord()
        record.id = 'test'
        record.dataArrayKey = 'test'
        const vectorItem = {
          'page': 0,
          'index': 0,
          'text': 'This is a test line.',
          'embedding': [0, 0, 0, 0, 0]
        }
        const vectorRecords = [vectorItem]
        const result3 = await record.setVectors(vectorRecords)
        const check3 = result3 != ['test']
        console.log(check3)
      }

        return 'Database tests complete'
      }
    async createAllStoresInDb() {
      //clear any previous db
      await deleteDB(this.dbName, {
        blocked() {
          console.log(`error: can't delete ${this.dbName} because there are open connections`)
        },
      })
      const checks = []
      await this.createStoreInDb(this.dbName, this.dbVersion, this.storeNamesAndKeyFields, checks)
      return checks
    }
    async createStoreInDb(dbName, dbVersion, storeNamesAndKeyFields, checks) {
      //create new db
      const dbPromise = await openDB(dbName, dbVersion, {
        upgrade(db) {
          for (const item of storeNamesAndKeyFields) {
            db.createObjectStore(
              item.storeName, 
              { 
                keyPath: item.KeyField,
                autoIncrement: false 
              }
            );
            console.log(`database created: ${item.storeName}`)
            checks.push(true)
          }
          return checks
        }
      })
    }
}


export async function getIdbAvalabileSpace(){
    // Get space available for IndexedDb
    //notes: The StorageManager isn't implemented in all browsers yet, so you must feature detect it before using it. Even when it is available, you must still catch over-quota errors (see below). In some cases, it's possible for the available quota to exceed the actual amount of storage available.
    //ref: https://web.dev/articles/storage-for-the-web#:~:text=In%20many%20browsers%2C%20you%20can%20use%20the%20StorageManager,to%20calculate%20the%20approximate%20remaining%20storage%20space%20available.
    if (navigator.storage && navigator.storage.estimate) {
        const quota = await navigator.storage.estimate();
        // quota.usage -> Number of bytes used.
        // quota.quota -> Maximum number of bytes available.
        const percentageUsed = (quota.usage / quota.quota) * 100;
        console.log(`You've used ${percentageUsed}% of the available storage.`);
        const remaining = quota.quota - quota.usage;
        console.log(`You can write up to ${getFormattedFileSize(remaining)} more bytes.`);
      }
}
  

// Set
export async function updateItemsInStore(dbName, dbVersion, storeName, arrayRecords){
  const db = await openDB(dbName, dbVersion);
  const tx = db.transaction(storeName, 'readwrite');

  // Update multiple items in the store in a single transaction:
    const tasks = []
    for(const record of arrayRecords){
        //tasks.push( tx.store.put(record) )
      if (storeName=='Document'){
        tasks.push( tx.store.put(record.dataArray, record.dataArrayKey) )
      }else if (storeName=='Vector'){
        tasks.push(tx.store.put({ record }, record.dataVectorKey))
      }
    }
    const check = await Promise.all(tasks)
    tx.done
    return check
    /*
    tx.store.put({
      name: 'Sandwich',
      price: 5.99,
      description: 'A MORE tasty sandwich!',
      updated: new Date().getTime() // This creates a new field
    }),
    tx.store.put({
      name: 'Eggs',
      price: 3.99,
      description: 'Some even NICER eggs you can cook up!',
      updated: new Date().getTime() // This creates a new field
    }),
    tx.done
  ]);*/
}


// Get
export async function getItemFromStore(dbName, dbVersion, storeName, key) {
    const db = await openDB(dbName, dbVersion);
    const value = await db.get(storeName, key);
    return value
  }
  